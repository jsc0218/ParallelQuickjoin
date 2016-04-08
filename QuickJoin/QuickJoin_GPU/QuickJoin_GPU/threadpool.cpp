#include <stdlib.h>
#include <pthread.h>
#include "threadpool.h"

/**
 *  @struct threadpool_task
 *  @brief the work struct
 *  @var function Pointer to the function that will perform the task.
 *  @var argument Argument to be passed to the function.
 */
typedef struct {
    void (*function)(void *);
    void *argument;
} threadpool_task_t;

/**
 *  @struct threadpool
 *  @brief The threadpool struct
 *  @var notify       Condition variable to notify worker threads.
 *  @var threads      Array containing worker threads ID.
 *  @var thread_count Number of threads
 *  @var queue        Array containing the task queue.
 *  @var queue_size   Size of the task queue.
 *  @var head         Index of the first element.
 *  @var tail         Index of the next element.
 *  @var shutdown     Flag indicating if the pool is shutting down
 */
struct threadpool_t {
  pthread_mutex_t lock;
  pthread_cond_t notify;
  pthread_t *threads;
  int thread_count;
  threadpool_task_t *queue;
  int queue_size;
  int head;
  int tail;
  int count;
  int shutdown;
  int started;
  int working;
};

/**
 * @function void *threadpool_thread(void *threadpool)
 * @brief the worker thread
 * @param threadpool the pool which own the thread
 */
void *threadpool_thread(void *threadpool)
{
    threadpool_t *pool = (threadpool_t *)threadpool;

    while (1) 
	{
        pthread_mutex_lock(&(pool->lock));

		--(pool->working);

        while (pool->count==0 && !(pool->shutdown))
		{
            pthread_cond_wait(&(pool->notify), &(pool->lock));
        }

        if (pool->shutdown)
		{
            break;
        }

		threadpool_task_t task;
        task.function = pool->queue[pool->head].function;
        task.argument = pool->queue[pool->head].argument;
        ++(pool->head);
        pool->head = (pool->head == pool->queue_size) ? 0 : pool->head;
        --(pool->count);

		++(pool->working);

        pthread_mutex_unlock(&(pool->lock));

        (*(task.function))(task.argument);
    }

    --(pool->started);

    pthread_mutex_unlock(&(pool->lock));
    
	pthread_exit(NULL);
    return NULL;
}

int threadpool_free(threadpool_t *pool)
{
	if (pool==NULL || pool->started>0)
	{
		return -1;
	}

	if (pool->threads) 
	{
		free(pool->threads);
	}
	if (pool->queue)
	{
		free(pool->queue);
	}
	pthread_mutex_destroy(&(pool->lock));
	pthread_cond_destroy(&(pool->notify));
	free(pool);    
	return 0;
}

threadpool_t *threadpool_create(int thread_count, int queue_size)
{
    threadpool_t *pool = (threadpool_t *)malloc(sizeof(threadpool_t));
    if (pool == NULL) 
	{
        goto err;
    }

	if (pthread_mutex_init(&(pool->lock),NULL) != 0)
	{
		free(pool);
		return NULL;
	}
	if (pthread_cond_init(&(pool->notify),NULL) != 0)
	{
		pthread_mutex_destroy(&(pool->lock));
		free(pool);
		return NULL;
	}
    pool->thread_count = thread_count;
    pool->queue_size = queue_size;
    pool->head = pool->tail = pool->count = 0;
    pool->shutdown = pool->started = 0;
	pool->working = thread_count;
    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * thread_count);
    pool->queue = (threadpool_task_t *)malloc(sizeof(threadpool_task_t) * queue_size);
    if (pool->threads==NULL || pool->queue==NULL) 
	{
        goto err;
    }

    for (int i=0; i<thread_count; ++i)
	{
        if (pthread_create(&(pool->threads[i]),NULL,threadpool_thread,(void*)pool) != 0) 
		{
            threadpool_destroy(pool);
            return NULL;
        }
		else 
		{
            ++(pool->started);
        }
    }

    return pool;

 err:
    if (pool) 
	{
        threadpool_free(pool);
    }
    return NULL;
}

int threadpool_add(threadpool_t *pool, void (*function)(void *), void *argument)
{
    if (pool==NULL || function==NULL)
	{
        return threadpool_invalid;
    }

    if (pthread_mutex_lock(&(pool->lock)) != 0) 
	{
        return threadpool_lock_failure;
    }

    int next = pool->tail + 1;
    next = (next == pool->queue_size) ? 0 : next;
	
	int err = 0;
    do 
	{
        if (pool->count == pool->queue_size) 
		{
            err = threadpool_queue_full;
            break;
        }

        if (pool->shutdown) 
		{
            err = threadpool_shutdown;
            break;
        }

        pool->queue[pool->tail].function = function;
        pool->queue[pool->tail].argument = argument;
        pool->tail = next;
        ++(pool->count);

        if (pthread_cond_signal(&(pool->notify)) != 0) 
		{
            err = threadpool_lock_failure;
            break;
        }
    } while (0);

    if (pthread_mutex_unlock(&pool->lock) != 0) 
	{
        err = threadpool_lock_failure;
    }

    return err;
}

int threadpool_destroy(threadpool_t *pool)
{
    if (pool == NULL) 
	{
        return threadpool_invalid;
    }

    if (pthread_mutex_lock(&(pool->lock)) != 0)
	{
        return threadpool_lock_failure;
    }

	int err = 0;
    do 
	{
        if (pool->shutdown)
		{
            err = threadpool_shutdown;
            break;
        }
        pool->shutdown = 1;

        if (pthread_cond_broadcast(&(pool->notify))!=0 || pthread_mutex_unlock(&(pool->lock))!=0) 
		{
            err = threadpool_lock_failure;
            break;
        }

        for (int i=0; i<pool->thread_count; ++i)
		{
            if (pthread_join(pool->threads[i], NULL) != 0) 
			{
                err = threadpool_thread_failure;
            }
        }
    } while (0);

    if (pthread_mutex_unlock(&pool->lock) != 0)
	{
        err = threadpool_lock_failure;
    }
    
    if (!err)
	{
        threadpool_free(pool);
    }
    return err;
}

bool threadpool_destroy_ready(threadpool_t *pool)
{
	bool res = true;
	pthread_mutex_lock(&(pool->lock));
	if (pool->count>0 || pool->working>0)
	{
		res = false;
	}
	pthread_mutex_unlock(&(pool->lock));
	return res;
}

