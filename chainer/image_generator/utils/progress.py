import sys

class print_progress():
    def __init__(self, max_iter, batch_size, data_size):
        self._max_iter = max_iter
        self.batch_size = batch_size
        self.data_size = data_size
        self.total_time = 0
    
    def __call__(self, elapsed, state):
        self.total_time += elapsed
        now_epoch, now_iter, now_loc = state

        iter_per = now_iter / self._max_iter
        n_sharp = int(50 * iter_per)
        n_dot = 50 - n_sharp
        log_text = "total [" + "#"*n_sharp + "."*n_dot + f"] {iter_per:.2%}\n"

        data_per = now_loc / self.data_size
        n_sharp = int(50 * data_per)
        n_dot = 50 - n_sharp
        log_text += "iter  [" + "#"*n_sharp + "."*n_dot + f"] {data_per:.2%}\n"
        log_text += f"{now_iter} iter, {now_epoch} epoch / {self._max_iter} iterations\n"

        est_total_time = (self.total_time / now_iter) * self._max_iter - self.total_time
        if est_total_time < 0:est_total_time = 0.0
        day = int((est_total_time) / (24 * 60 * 60))
        est_total_time -= day * 24 * 60 * 60
        hour = int(est_total_time / (60 * 60))
        est_total_time -= hour * 60 * 60
        min = int(est_total_time /  60)
        sec = est_total_time - min * 60

        time = f"{hour:02d}:{min:02d}:{sec:.4f}"
        iter_per_sec = now_iter / self.total_time
        log_text += f"{iter_per_sec:.4f} iter/sec, Estimated time to finishe: {day} day, {time}"

        print(log_text)
        print("\u001B[4A", end="")
    
    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, new_max_iter):
        self._max_iter = new_max_iter


def progress(count, file_count):
    percentage = count / file_count
    n_sharp = int(50 * percentage)
    n_dot = 50 - n_sharp
    log_text = "progress  [" + "#"*n_sharp + "."*n_dot + "] "
    sys.stdout.write(log_text + f"{percentage:.2%} ( {count} / {file_count} )\r")