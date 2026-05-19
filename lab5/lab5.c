#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TASKS_PER_PROC  100       
#define WEIGHT_SCALE    4000000    
#define NUM_ITERATIONS  5         
#define TASK_BUF_CAP    (TASKS_PER_PROC * ( 64 + 1))

#define TAG_SERVER_CMD  10   
#define TAG_REPLY       11

/* ── Структура задания ─────────────────────────────────────── */
typedef struct {
    int repeatNum;  /* сколько раз считать sqrt — вычислительный вес */
} Task;

/* ── Глобальные переменные процесса ────────────────────────── */
static int rank, size;

/* --- Буфер задач --- */
static Task task_buf[TASK_BUF_CAP];
static int  task_head = 0;   /* индекс следующего задания для выполнения */
static int  task_tail = 0;   /* индекс за последним имеющимся заданием   */
static pthread_mutex_t task_mtx = PTHREAD_MUTEX_INITIALIZER;

/* --- Канал worker ↔ requester --- */
/*
 * worker выставляет req_needed=1 и ждёт на req_answered.
 * requester просыпается, ищет работу, выставляет req_found и
 * req_needed=0, сигналит req_answered.
 */
static int req_needed = 0;  /* worker просит работу                */
static int req_found  = 0;  /* requester нашёл (1) / не нашёл (0)  */
static int shutdown   = 0;  /* флаг завершения для requester        */
static pthread_mutex_t req_mtx     = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  req_needed_cond  = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  req_answered_cond = PTHREAD_COND_INITIALIZER;

/* --- Результат вычислений --- */
static double global_res = 0.0;
static pthread_mutex_t res_mtx = PTHREAD_MUTEX_INITIALIZER;

/* --- Отладочные счётчики --- */
static int tasks_done_iter  = 0;  /* задач выполнено за текущую итерацию  */
static int tasks_done_total = 0;  /* задач выполнено за всё время работы   */
static pthread_mutex_t dbg_mtx = PTHREAD_MUTEX_INITIALIZER;

/* ══════════════════════════════════════════════════════════════
 * Работа с буфером задач
 * ══════════════════════════════════════════════════════════════ */

/* Инициализировать буфер для новой итерации.
 * Вес задания ∝ abs(rank - iter%size) → «волна» нагрузки. */
static void tasks_init(int iter)
{
    pthread_mutex_lock(&task_mtx);
    int wave = abs(rank - (iter % size));
    for (int i = 0; i < TASKS_PER_PROC; i++) {
        int w = (wave + 1) * WEIGHT_SCALE;
        /* небольшая случайность, чтобы задания отличались */
        w += rand() % (WEIGHT_SCALE / 5 + 1);
        if (w < 1) w = 1;
        task_buf[i].repeatNum = w;
    }
    task_head = 0;
    task_tail = TASKS_PER_PROC;
    pthread_mutex_unlock(&task_mtx);
}

/* Взять одно задание. Возвращает 1 при успехе, 0 если пусто. */
static int tasks_fetch(Task *out)
{
    pthread_mutex_lock(&task_mtx);
    int ok = (task_head < task_tail);
    if (ok) *out = task_buf[task_head++];
    pthread_mutex_unlock(&task_mtx);
    return ok;
}

/* Отдать половину оставшихся заданий в out[].
 * Возвращает количество отданных (0 если нечего). */
static int tasks_give_half(Task *out, int out_cap)
{
    pthread_mutex_lock(&task_mtx);
    int remaining = task_tail - task_head;
    int give = remaining / 2;
    if (give > out_cap) give = out_cap;
    if (give > 0) {
        task_tail -= give;
        memcpy(out, task_buf + task_tail, give * sizeof(Task));
    }
    pthread_mutex_unlock(&task_mtx);
    return give;
}

/* Добавить n задач из src[] в конец буфера. */
static void tasks_append(const Task *src, int n)
{
    pthread_mutex_lock(&task_mtx);
    int free_space = TASK_BUF_CAP - task_tail;
    if (n > free_space) n = free_space;
    if (n > 0) {
        memcpy(task_buf + task_tail, src, n * sizeof(Task));
        task_tail += n;
    }
    pthread_mutex_unlock(&task_mtx);
}

/* ══════════════════════════════════════════════════════════════
 * Поток-сервер: принимает запросы о работе от других процессов
 * ══════════════════════════════════════════════════════════════ */
static void *server_thread(void *arg)
{
    (void)arg;
    Task reply_buf[TASKS_PER_PROC];

    for (;;) {
        MPI_Status st;
        int cmd; // 1 = REQUEST, 0 = STOP
        
        // Убрали MPI_ANY_TAG, теперь ловим только TAG_SERVER_CMD
        MPI_Recv(&cmd, 1, MPI_INT,
                 MPI_ANY_SOURCE, TAG_SERVER_CMD, MPI_COMM_WORLD, &st);

        if (cmd == 0) // Это сигнал TAG_STOP
            break;  

        if (cmd == 1) { // Это сигнал TAG_REQUEST
            int give = tasks_give_half(reply_buf, TASKS_PER_PROC);
            MPI_Send(reply_buf,
                     give * (int)sizeof(Task), MPI_BYTE,
                     st.MPI_SOURCE, TAG_REPLY, MPI_COMM_WORLD);
        }
    }
    return NULL;
}

/* ══════════════════════════════════════════════════════════════
 * Поток-запросчик: ищет работу у других процессов по запросу
 * ══════════════════════════════════════════════════════════════ */
static void *requester_thread(void *arg)
{
    (void)arg;
    Task recv_buf[TASKS_PER_PROC];

    for (;;) {
        /* Засыпаем, пока worker не попросит работу */
        pthread_mutex_lock(&req_mtx);
        while (!req_needed && !shutdown)
            pthread_cond_wait(&req_needed_cond, &req_mtx);

        if (shutdown) {
            pthread_mutex_unlock(&req_mtx);
            break;
        }
        pthread_mutex_unlock(&req_mtx);

        /* Обходим все остальные процессы по кругу начиная с rank+1 */
        int found = 0;
        for (int step = 1; step < size && !found; step++) {
            int target = (rank + step) % size;
            int dummy = 0;
            int cmd = 1; // Запрашиваем работу
            MPI_Send(&cmd, 1, MPI_INT, target, TAG_SERVER_CMD, MPI_COMM_WORLD);
            MPI_Status st;
            MPI_Recv(recv_buf,
                     TASKS_PER_PROC * (int)sizeof(Task), MPI_BYTE,
                     target, TAG_REPLY, MPI_COMM_WORLD, &st);

            int bytes;
            MPI_Get_count(&st, MPI_BYTE, &bytes);
            int n = bytes / (int)sizeof(Task);

            if (n > 0) {
                tasks_append(recv_buf, n);
                printf("[rank %d] STOLE %d tasks from rank %d\n", rank, n, target);
                fflush(stdout); // Чтобы текст сразу выводился в консоль
                found = 1;
            }
        }

        /* Сообщаем worker результат */
        pthread_mutex_lock(&req_mtx);
        req_needed = 0;
        req_found  = found;
        pthread_cond_signal(&req_answered_cond);
        pthread_mutex_unlock(&req_mtx);
    }
    return NULL;
}



//Worker
static void run_iteration(void)
{
    for (;;) {
        Task t;
        if (tasks_fetch(&t)) {
            double res = 0.0;

            for (int i = 0; i < t.repeatNum; i++)
                res += sqrt((double)i);

            pthread_mutex_lock(&res_mtx);
            global_res += res;
            pthread_mutex_unlock(&res_mtx);
            pthread_mutex_lock(&dbg_mtx);
            tasks_done_iter++;
            tasks_done_total++;
            pthread_mutex_unlock(&dbg_mtx);
        } else {
            /* Локальный список пуст — просим запросчика найти работу */
            pthread_mutex_lock(&req_mtx);
            req_needed = 1;
            pthread_cond_signal(&req_needed_cond);
            /* Ждём ответа */
            while (req_needed)
                pthread_cond_wait(&req_answered_cond, &req_mtx);
            int found = req_found;
            pthread_mutex_unlock(&req_mtx);

            if (!found)
                break;  /* нигде нет работы — итерация завершена */
            /* нашли работу → продолжаем цикл fetch */
        }
    }
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        fprintf(stderr,
            "ERROR: MPI_THREAD_MULTIPLE not supported (provided=%d).\n"
            "Try: mpirun --mca mpi_thread_multiple 1 ...\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    srand((unsigned)(rank * 7919 + 1));

    pthread_t srv_tid, req_tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_create(&srv_tid, &attr, server_thread,    NULL);
    pthread_create(&req_tid, &attr, requester_thread, NULL);
    pthread_attr_destroy(&attr);

    /* ── Основной цикл итераций ── */
    double wall_start = MPI_Wtime();

    for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {
        tasks_init(iter);
        tasks_done_iter = 0;   /* сбрасываем счётчик итерации */

        /* ── НАЧАЛО ВОЛНЫ ── */
        pthread_mutex_lock(&task_mtx);
        int buf_size_at_start = task_tail - task_head;
        pthread_mutex_unlock(&task_mtx);
        int wave_val = abs(rank - (iter % size));
        printf("[rank %2d] ╔══ WAVE %d START ══╗  buf=%d tasks  wave_factor=%d\n",
               rank, iter, buf_size_at_start, wave_val);
        fflush(stdout);

        double t0 = MPI_Wtime();
        run_iteration();
        double t1 = MPI_Wtime();

        /* ── КОНЕЦ ВОЛНЫ ── */
        printf("[rank %2d] ╚══ WAVE %d END   ══╝  done_this_wave=%d  total_done=%d  iter_time=%.3f s\n",
               rank, iter, tasks_done_iter, tasks_done_total, t1 - t0);
        fflush(stdout);

        /* Барьер: все должны завершить итерацию перед началом следующей */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double wall_end = MPI_Wtime();

    /* ── Корректное завершение вспомогательных потоков ── */

    /* 1. Сервер: self-send с TAG_STOP */
   /* 1. Сервер: self-send с командой STOP */
    {
        int cmd = 0; // Команда на выключение
        MPI_Send(&cmd, 1, MPI_INT, rank, TAG_SERVER_CMD, MPI_COMM_WORLD);
    }

    /* 2. Запросчик: выставляем флаг shutdown и будим */
    pthread_mutex_lock(&req_mtx);
    shutdown = 1;
    pthread_cond_signal(&req_needed_cond);
    pthread_mutex_unlock(&req_mtx);

    pthread_join(srv_tid, NULL);
    pthread_join(req_tid,  NULL);

    /* ── Итоговая статистика каждого процесса ── */
    printf("[rank %2d] *** FINAL: total tasks executed = %d  global_res = %.6e  wall = %.3f s\n",
           rank, tasks_done_total, global_res, wall_end - wall_start);
    fflush(stdout);

    /* ── Сбор суммарного результата со всех процессов ── */
    double total_result = 0.0;
    MPI_Reduce(&global_res, &total_result, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Summary ===\n");
        printf("Processes  : %d\n", size);
        printf("Iterations : %d\n", NUM_ITERATIONS);
        printf("Sum result : %.6e\n", total_result);
        printf("Wall time  : %.3f sec\n", wall_end - wall_start);
    }

    MPI_Finalize();
    return 0;
}
