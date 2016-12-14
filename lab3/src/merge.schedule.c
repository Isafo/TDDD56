#include <stdlib.h> 

#include <drake/schedule.h> 

#include <drake/platform.h> 

#define TASK_NAME leaf_1
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_1 1

#define TASK_NAME leaf_2
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_2 1

#define TASK_NAME leaf_3
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_3 1

#define TASK_NAME leaf_4
#define TASK_MODULE presort
#include <drake/node.h>
#define DONE_leaf_4 1

#define TASK_NAME root
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_root 1

#define TASK_NAME root_1
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_root_1 1

#define TASK_NAME root_2
#define TASK_MODULE merge
#include <drake/node.h>
#define DONE_root_2 1

int drake_task_number()
{
	return 7;
}

char* drake_task_name(size_t index)
{
	switch(index - 1)
	{
		case 0:
			return "leaf_1";
		break;
		case 1:
			return "leaf_2";
		break;
		case 2:
			return "leaf_3";
		break;
		case 3:
			return "leaf_4";
		break;
		case 4:
			return "root";
		break;
		case 5:
			return "root_1";
		break;
		case 6:
			return "root_2";
		break;
		default:
			return "invalid task id";
		break;
	}
}

void drake_schedule_init(drake_schedule_t* schedule)
{
	schedule->core_number = 4;
	schedule->task_number = 7;
	schedule->stage_time = 0;

	schedule->tasks_in_core = malloc(sizeof(size_t) * schedule->core_number);

	schedule->task_name = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->task_name[0] = "leaf_1";
	schedule->task_name[1] = "leaf_2";
	schedule->task_name[2] = "leaf_3";
	schedule->task_name[3] = "leaf_4";
	schedule->task_name[4] = "root";
	schedule->task_name[5] = "root_1";
	schedule->task_name[6] = "root_2";


	schedule->task_workload = malloc(sizeof(double) * schedule->task_number);
	schedule->task_workload[0] = 1;
	schedule->task_workload[1] = 1;
	schedule->task_workload[2] = 1;
	schedule->task_workload[3] = 1;
	schedule->task_workload[4] = 1;
	schedule->task_workload[5] = 1;
	schedule->task_workload[6] = 1;

	schedule->tasks_in_core[0] = 2;
	schedule->tasks_in_core[1] = 1;
	schedule->tasks_in_core[2] = 2;
	schedule->tasks_in_core[3] = 2;

	schedule->consumers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->consumers_in_core[0] = 1;
	schedule->consumers_in_core[1] = 1;
	schedule->consumers_in_core[2] = 1;
	schedule->consumers_in_core[3] = 1;

	schedule->producers_in_core = malloc(sizeof(size_t) * schedule->core_number);
	schedule->producers_in_core[0] = 1;
	schedule->producers_in_core[1] = 0;
	schedule->producers_in_core[2] = 2;
	schedule->producers_in_core[3] = 1;

	schedule->consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->consumers_in_task[0] = 1;
	schedule->consumers_in_task[1] = 1;
	schedule->consumers_in_task[2] = 1;
	schedule->consumers_in_task[3] = 1;
	schedule->consumers_in_task[4] = 0;
	schedule->consumers_in_task[5] = 1;
	schedule->consumers_in_task[6] = 1;

	schedule->producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->producers_in_task[0] = 0;
	schedule->producers_in_task[1] = 0;
	schedule->producers_in_task[2] = 0;
	schedule->producers_in_task[3] = 0;
	schedule->producers_in_task[4] = 2;
	schedule->producers_in_task[5] = 2;
	schedule->producers_in_task[6] = 2;

	schedule->remote_consumers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_consumers_in_task[0] = 0;
	schedule->remote_consumers_in_task[1] = 1;
	schedule->remote_consumers_in_task[2] = 1;
	schedule->remote_consumers_in_task[3] = 0;
	schedule->remote_consumers_in_task[4] = 0;
	schedule->remote_consumers_in_task[5] = 1;
	schedule->remote_consumers_in_task[6] = 1;

	schedule->remote_producers_in_task = malloc(sizeof(size_t) * schedule->task_number);
	schedule->remote_producers_in_task[0] = 0;
	schedule->remote_producers_in_task[1] = 0;
	schedule->remote_producers_in_task[2] = 0;
	schedule->remote_producers_in_task[3] = 0;
	schedule->remote_producers_in_task[4] = 2;
	schedule->remote_producers_in_task[5] = 1;
	schedule->remote_producers_in_task[6] = 1;

	schedule->producers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->producers_rate = malloc(sizeof(size_t**) * schedule->task_number);
	schedule->producers_name = malloc(sizeof(char*) * schedule->task_number);
	schedule->producers_id[0] = NULL;
	schedule->producers_rate[0] = NULL;
	schedule->producers_name[0] = NULL;
	schedule->producers_id[1] = NULL;
	schedule->producers_rate[1] = NULL;
	schedule->producers_name[1] = NULL;
	schedule->producers_id[2] = NULL;
	schedule->producers_rate[2] = NULL;
	schedule->producers_name[2] = NULL;
	schedule->producers_id[3] = NULL;
	schedule->producers_rate[3] = NULL;
	schedule->producers_name[3] = NULL;
	schedule->producers_id[4] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[4][0] = 6;
	schedule->producers_id[4][1] = 7;
	schedule->producers_rate[4] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[4][0] = 1;
	schedule->producers_rate[4][1] = 1;
	schedule->producers_name[4] = malloc(sizeof(char*) * 2);
	schedule->producers_name[4][0] = "output";
	schedule->producers_name[4][1] = "output";
	schedule->producers_id[5] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[5][0] = 1;
	schedule->producers_id[5][1] = 2;
	schedule->producers_rate[5] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[5][0] = 1;
	schedule->producers_rate[5][1] = 1;
	schedule->producers_name[5] = malloc(sizeof(char*) * 2);
	schedule->producers_name[5][0] = "output";
	schedule->producers_name[5][1] = "output";
	schedule->producers_id[6] = malloc(sizeof(size_t) * 2);
	schedule->producers_id[6][0] = 3;
	schedule->producers_id[6][1] = 4;
	schedule->producers_rate[6] = malloc(sizeof(size_t) * 2);
	schedule->producers_rate[6][0] = 1;
	schedule->producers_rate[6][1] = 1;
	schedule->producers_name[6] = malloc(sizeof(char*) * 2);
	schedule->producers_name[6][0] = "output";
	schedule->producers_name[6][1] = "output";

	schedule->consumers_id = malloc(sizeof(size_t*) * schedule->task_number);
	schedule->consumers_rate = malloc(sizeof(size_t**) * schedule->task_number);
	schedule->consumers_name = malloc(sizeof(char*) * schedule->task_number);
	schedule->consumers_id[0] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[0][0] = 6;
	schedule->consumers_rate[0] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[0][0] = 1;
	schedule->consumers_name[0] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[0][0] = "left";
	schedule->consumers_id[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[1][0] = 6;
	schedule->consumers_rate[1] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[1][0] = 1;
	schedule->consumers_name[1] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[1][0] = "right";
	schedule->consumers_id[2] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[2][0] = 7;
	schedule->consumers_rate[2] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[2][0] = 1;
	schedule->consumers_name[2] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[2][0] = "left";
	schedule->consumers_id[3] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[3][0] = 7;
	schedule->consumers_rate[3] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[3][0] = 1;
	schedule->consumers_name[3] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[3][0] = "right";
	schedule->consumers_id[4] = NULL;
	schedule->consumers_rate[4] = NULL;
	schedule->consumers_name[4] = NULL;
	schedule->consumers_id[5] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[5][0] = 5;
	schedule->consumers_rate[5] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[5][0] = 1;
	schedule->consumers_name[5] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[5][0] = "right";
	schedule->consumers_id[6] = malloc(sizeof(size_t) * 1);
	schedule->consumers_id[6][0] = 5;
	schedule->consumers_rate[6] = malloc(sizeof(size_t) * 1);
	schedule->consumers_rate[6][0] = 1;
	schedule->consumers_name[6] = malloc(sizeof(char*) * 1);
	schedule->consumers_name[6][0] = "left";

	schedule->schedule = malloc(sizeof(drake_schedule_task_t*) * schedule->core_number);
	schedule->schedule[0] = malloc(sizeof(drake_schedule_task_t) * 2);
	schedule->schedule[0][0].id = 1;
	schedule->schedule[0][0].start_time = 2.86e-07;
	schedule->schedule[0][0].frequency = 16;
	schedule->schedule[0][1].id = 6;
	schedule->schedule[0][1].start_time = 5.71e-07;
	schedule->schedule[0][1].frequency = 16;
	schedule->schedule[1] = malloc(sizeof(drake_schedule_task_t) * 1);
	schedule->schedule[1][0].id = 2;
	schedule->schedule[1][0].start_time = 0;
	schedule->schedule[1][0].frequency = 16;
	schedule->schedule[2] = malloc(sizeof(drake_schedule_task_t) * 2);
	schedule->schedule[2][0].id = 3;
	schedule->schedule[2][0].start_time = 0;
	schedule->schedule[2][0].frequency = 16;
	schedule->schedule[2][1].id = 5;
	schedule->schedule[2][1].start_time = 5.71e-07;
	schedule->schedule[2][1].frequency = 16;
	schedule->schedule[3] = malloc(sizeof(drake_schedule_task_t) * 2);
	schedule->schedule[3][0].id = 4;
	schedule->schedule[3][0].start_time = 0;
	schedule->schedule[3][0].frequency = 16;
	schedule->schedule[3][1].id = 7;
	schedule->schedule[3][1].start_time = 5.71e-07;
	schedule->schedule[3][1].frequency = 16;
}

void drake_schedule_destroy(drake_schedule_t* schedule)
{
	free(schedule->schedule[0]);
	free(schedule->schedule[1]);
	free(schedule->schedule[2]);
	free(schedule->schedule[3]);

	free(schedule->schedule);
	free(schedule->consumers_id[0]);
	if(schedule->consumers_rate[0] != NULL)
	{
		free(schedule->consumers_rate[0]);
	}
	free(schedule->consumers_name[0]);
	free(schedule->consumers_id[1]);
	if(schedule->consumers_rate[1] != NULL)
	{
		free(schedule->consumers_rate[1]);
	}
	free(schedule->consumers_name[1]);
	free(schedule->consumers_id[2]);
	if(schedule->consumers_rate[2] != NULL)
	{
		free(schedule->consumers_rate[2]);
	}
	free(schedule->consumers_name[2]);
	free(schedule->consumers_id[3]);
	if(schedule->consumers_rate[3] != NULL)
	{
		free(schedule->consumers_rate[3]);
	}
	free(schedule->consumers_name[3]);
	free(schedule->consumers_id[4]);
	if(schedule->consumers_rate[4] != NULL)
	{
		free(schedule->consumers_rate[4]);
	}
	free(schedule->consumers_name[4]);
	free(schedule->consumers_id[5]);
	if(schedule->consumers_rate[5] != NULL)
	{
		free(schedule->consumers_rate[5]);
	}
	free(schedule->consumers_name[5]);
	free(schedule->consumers_id[6]);
	if(schedule->consumers_rate[6] != NULL)
	{
		free(schedule->consumers_rate[6]);
	}
	free(schedule->consumers_name[6]);
	free(schedule->consumers_id);
	free(schedule->consumers_rate);
	free(schedule->consumers_name);

	free(schedule->producers_id[0]);
	if(schedule->producers_rate[0] != NULL)
	{
		free(schedule->producers_rate[0]);
	}
	free(schedule->producers_name[0]);
	free(schedule->producers_id[1]);
	if(schedule->producers_rate[1] != NULL)
	{
		free(schedule->producers_rate[1]);
	}
	free(schedule->producers_name[1]);
	free(schedule->producers_id[2]);
	if(schedule->producers_rate[2] != NULL)
	{
		free(schedule->producers_rate[2]);
	}
	free(schedule->producers_name[2]);
	free(schedule->producers_id[3]);
	if(schedule->producers_rate[3] != NULL)
	{
		free(schedule->producers_rate[3]);
	}
	free(schedule->producers_name[3]);
	free(schedule->producers_id[4]);
	if(schedule->producers_rate[4] != NULL)
	{
		free(schedule->producers_rate[4]);
	}
	free(schedule->producers_name[4]);
	free(schedule->producers_id[5]);
	if(schedule->producers_rate[5] != NULL)
	{
		free(schedule->producers_rate[5]);
	}
	free(schedule->producers_name[5]);
	free(schedule->producers_id[6]);
	if(schedule->producers_rate[6] != NULL)
	{
		free(schedule->producers_rate[6]);
	}
	free(schedule->producers_name[6]);
	free(schedule->producers_id);
	free(schedule->producers_rate);
	free(schedule->producers_name);
	free(schedule->task_workload);
	free(schedule->remote_producers_in_task);
	free(schedule->remote_consumers_in_task);
	free(schedule->producers_in_task);
	free(schedule->consumers_in_task);
	free(schedule->producers_in_core);
	free(schedule->consumers_in_core);
	free(schedule->tasks_in_core);
	free(schedule->task_name);
}

size_t
drake_task_width(task_tp task){
	size_t task_width[7] = {1, 1, 1, 1, 1, 1, 1, };
	return task_width[task->id - 1];
}

size_t
drake_core_id(task_tp task){
	size_t local_core_id[4][7] = {
			{0, 4, 4, 4, 4, 0, 4, },
			{4, 0, 4, 4, 4, 4, 4, },
			{4, 4, 0, 4, 0, 4, 4, },
			{4, 4, 4, 0, 4, 4, 0, },
		};
	return local_core_id[drake_platform_core_id()][task->id - 1];
}

void*
drake_function(size_t id, task_status_t status)
{
	switch(id)
	{
		default:
			// TODO: Raise an alert
		break;
		case 1:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_1);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_1);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_1);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_1);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_1);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 2:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_2);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_2);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_2);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_2);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_2);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 3:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_3);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_3);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_3);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_3);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_3);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 4:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(presort, leaf_4);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(presort, leaf_4);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(presort, leaf_4);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(presort, leaf_4);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(presort, leaf_4);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 5:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, root);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, root);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, root);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, root);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, root);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 6:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, root_1);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, root_1);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, root_1);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, root_1);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, root_1);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;
		case 7:
			switch(status)
			{
				case TASK_INVALID:
					return 0;
				break;
				case TASK_INIT:
					return (void*)&drake_init(merge, root_2);
 				break;
  				case TASK_START:
  					return (void*)&drake_start(merge, root_2);
  				break;
  				case TASK_RUN:
  					return (void*)&drake_run(merge, root_2);
  				break;
  				case TASK_KILLED:
  					return (void*)&drake_kill(merge, root_2);
  				break;
  				case TASK_ZOMBIE:
  					return 0;
  				break;
  				case TASK_DESTROY:
  					return (void*)&drake_destroy(merge, root_2);
  				break;
  				default:
  					return 0;
  				break;
  			}
  		break;

	}

	return 0;
}
