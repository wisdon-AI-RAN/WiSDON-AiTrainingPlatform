#==========================================================
# Description: Task queue for AI Training Platform.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2026/02/06
# Version: 0.1.0
# License: None
#==========================================================

import queue

class TrainingTaskQueue:
    def __init__(self, logger):
        """
        Define variable in the task queue.
        """
        self.logger = logger

        # Create queue to record waited tasks which need to restart reinforcement/federated learning
        self.fl_task_queue = queue.Queue()
        # Record current reinforcement/federated learning task
        self.fl_current_training_task = None
        # Record each task's training participants
        # self.fl_training_participants = dict()
        # Create list to record tasks which need to delete
        self.fl_training_delete_list = list()

    def fl_training_queue_push(self, task_name: str, client_id: str):
        """
        Define functions called by api functions.
        Push the task which need to process the federated learning event into the queue.
        """
        # Add client to task's participant
        # if task_name not in self.fl_training_participants:
        #     self.fl_training_participants[task_name] = set()
        # self.fl_training_participants[task_name].add(client_id)
        # Check whether the same task in the queue
        if not self.is_duplicate_report(task_name):
            self.fl_task_queue.put(task_name)
        self.logger.info(f'FL Training Event Push : {task_name} !')

    def fl_training_queue_pop(self) -> bool:
        """
        Define functions run with api concurrently.
        Pop the task which first ready to process the federated learning event.
        """
        try:
            if self.fl_current_training_task != None:
                return True
            self.fl_current_training_task = self.fl_task_queue.get(timeout=3)
            # Check whether the task need to delete
            if self.fl_current_training_task in self.fl_training_delete_list:
                self.fl_training_delete_list.remove(self.fl_current_training_task)
                # del self.fl_training_participants[self.fl_current_training_task]
                self.logger.info(f'FL Training Event Delete : {self.fl_current_training_task} !')
                self.fl_current_training_task = None
                return True
            self.logger.info(f'FL Training Event Trigger : {self.fl_current_training_task} !')
            return True
        except queue.Empty:
            return False
        
    def fl_training_queue_delete(self, task_name: str) -> bool:
        """
        Define functions called by api functions.
        Delete the current or waiting federated learning task.
        """
        # Check whether the task need to delete
        if self.fl_current_training_task == task_name:
            self.logger.info(f'FL Training Event Delete : {self.fl_current_training_task} !')
            # del self.fl_training_participants[self.fl_current_training_task]
            self.fl_current_training_task = None
        elif task_name not in self.fl_training_delete_list:
            self.fl_training_delete_list.append(task_name)
        return True
        
    def fl_training_queue_task_finish(self):
        """
        Define functions called by api functions.
        Finish the current federated learning task.
        """
        if self.fl_current_training_task:
            self.logger.info(f'FL Training Event Finished : {self.fl_current_training_task} !')
            # del self.fl_training_participants[self.fl_current_training_task]
            self.fl_current_training_task = None

    def is_duplicate_report(self, task_name: str) -> bool:
        """
        Define utils functions called by api functions.
        """
        with self.fl_task_queue.mutex:
            return (self.fl_current_training_task == task_name or any(queue_task == task_name for queue_task in self.fl_task_queue.queue))

    def get_queue_content(self, q : queue.Queue):
        """
        Define utils functions called by api functions.
        """
        with q.mutex:
            return list(q.queue)
        