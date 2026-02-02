#==========================================================
# Description: Task queue for AI Training Platform.
# =========================================================
# Author: Benson Jao (WiSDON)
# Date: 2025/09/15
# Version: 0.1.0
# License: None
#==========================================================

import queue

class TaskQueue:
    def __init__(self, logger):
        """
        Define global variable in api server.
        """
        self.logger = logger
        self.state = 0
        ''' retrain '''
        # Create queue to record waited tasks which need to restart reinforcement/federated learning
        self.fl_retrain_queue = queue.Queue()
        # Record current reinforcement/federated learning retrain task
        self.fl_retrain_task = None
        # Record flower server running status
        self.fl_server_status = False
        # Record each task's retrain participants
        self.fl_retrain_participants = dict()
        # Create list to record tasks which need to delete
        self.fl_retrain_delete_list = list()

        """
        Define retrain params set by AppPlat.
        """
        ''' training config '''
        # Record federated learning central round set by AppPlat
        self.fl_central_round = None
        # Record federated learning local epoch set by AppPlat
        self.fl_central_epoch = None

    def fl_retrain_queue_pop(self):
        """
        Define functions run with api concurrently.
        Pop the task which first ready to process the federated learning event.
        """
        while True:
            try:
                if self.fl_retrain_task != None:
                    continue
                self.fl_retrain_task = self.fl_retrain_queue.get(timeout=3)
                # Check whether the task need to delete
                if self.fl_retrain_task in self.fl_retrain_delete_list:
                    self.fl_retrain_delete_list.remove(self.fl_retrain_task)
                    self.logger.info(f'FL Retrain Event Delete : {self.fl_retrain_task} !')
                    self.fl_retrain_task = None
                    continue
                self.logger.info(f'FL Retrain Event Trigger : {self.fl_retrain_task} !')
            except queue.Empty:
                pass

    def is_duplicate_report(self, task_name: str) -> bool:
        """
        Define utils functions called by api functions.
        """
        with self.fl_retrain_queue.mutex:
            return (self.fl_retrain_task == task_name or any(queue_task == task_name for queue_task in self.fl_retrain_queue.queue))

    def get_queue_content(self, q : queue):
        """
        Define utils functions called by api functions.
        """
        with q.mutex:
            return list(q.queue)
        