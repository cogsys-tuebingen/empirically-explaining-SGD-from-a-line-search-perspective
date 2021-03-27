__author__ = ",  "
__version__ = "0.0"
__email__ = " "

import contextlib
import copy
import os
import pickle
import time

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class SGDLineSampler(Optimizer):
    """ Performs SGD training. For each update step direction the full-batch loss along the line in this direction is measured"""

    def __init__(self, model=required, lr=0.1, lr_milestones=[], momentum=0.0, l2_reg=0.0,
                 is_logging=True, is_plot=True, plot_save_dir="./lines/",
                 train_dataset=required, train_data_set_size=required, validation_dataset=required,
                 validation_data_set_size=required, batch_size=required, sample_interval=0.5, resolution=0.006,
                 max_interval_enlargements=5):

        assert 0 <= momentum < 1.0
        assert (train_data_set_size % batch_size) == 0
        assert (validation_data_set_size % batch_size) == 0

        params = model.parameters()
        self.model = model
        self.params = list(self.model.parameters())

        self.momentum = momentum
        self.sgd_lr = lr  # or scheduler
        self.lr_milestones = lr_milestones
        self.is_logging = is_logging
        self.is_plot = is_plot
        self.line_data_save_dir = plot_save_dir
        os.makedirs(self.line_data_save_dir, exist_ok=True)
        self.l2_reg = l2_reg
        self.epsilon = 1e-10

        self.direction_norm = torch.Tensor([1.0]).cuda()
        self.performed_training_steps = -1

        self.current_position = 0.0
        self.model_state_checkpoint = None
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        self.number_of_lines_measured = 0

        self.train_dataset = train_dataset
        self.train_data_set_size = train_data_set_size
        self.validation_dataset = validation_dataset
        self.validation_data_set_size = validation_data_set_size
        self.steps_per_train_epoch = train_data_set_size // batch_size
        self.steps_per_val_epoch = validation_data_set_size // batch_size
        self.batch_size = batch_size
        self.direction_batch_number = 0
        self.is_extrapolation = True

        self.sample_interval = sample_interval
        self.resolution = resolution
        self.max_interval_enlargements = max_interval_enlargements

        self.fist_update_call = True

        defaults = dict()  # only relevant for parameter groups which are not supported
        super(SGDLineSampler, self).__init__(params, defaults)

    def get_deterministic_loss_function(self, x, y, seed):
        def deterministic_loss():
            with self.random_seed_torch(int(seed)):
                logit = self.model(x.cuda())
                losses = self.loss(logit,
                                   y.cuda())
                return losses

        return deterministic_loss

    def measure_direction_batch_loss(self, direction_batch_loss_fn):

        with torch.no_grad():

            batch_size = self.batch_size

            sgd_update_step_location = self.update_step_size
            sample_interval = self.sample_interval
            resolution = self.resolution
            max_interval_enlargements = self.max_interval_enlargements

            all_measure_locations = []
            all_direction_batch_losses = []

            for remeasure in range(max_interval_enlargements):

                left_measure_locations = list(
                    np.arange(-sample_interval * (remeasure + 1), -sample_interval * remeasure, resolution))
                right_measure_locations = list(
                    np.arange(sample_interval * remeasure, sample_interval * (remeasure + 1), resolution))
                left_measure_locations.extend(right_measure_locations)
                measure_locations = left_measure_locations

                measure_locations.append(sgd_update_step_location)
                if 0.0 not in measure_locations:
                    measure_locations.append(0.0)
                sorted_measure_locations = list(np.sort(measure_locations))

                all_measure_locations.extend(sorted_measure_locations)
                direction_batch_losses = np.zeros((batch_size, len(sorted_measure_locations)))

                for i, location in enumerate(sorted_measure_locations):
                    step_size = location - self.current_position
                    self.current_position += step_size
                    self._perform_sgd_update(self.params, step_size)
                    _, dir_losses = direction_batch_loss_fn()
                    dir_losses = dir_losses.cpu().numpy()
                    direction_batch_losses[:, i] = dir_losses[:]

                all_direction_batch_losses.append(direction_batch_losses)

            sort_indexes = np.argsort(all_measure_locations)
            direction_batch_losses = np.hstack(all_direction_batch_losses)
            direction_batch_losses = direction_batch_losses[:, sort_indexes]

            path = self.line_data_save_dir + "/dir_batch_losses_data_{}.pickle".format(self.performed_training_steps)
            with open(path, 'wb') as f:
                pickle.dump((direction_batch_losses), f)

    def measure_exact_loss(self, direction_batch_loss_fn, direction_batch_number):

        with torch.no_grad():

            batch_size = self.batch_size
            train_dataset_size = self.train_data_set_size
            train_dataset = self.train_dataset
            train_steps_per_epoch = self.steps_per_train_epoch

            val_dataset_size = self.validation_data_set_size
            val_dataset = self.validation_dataset
            val_steps_per_epoch = self.steps_per_val_epoch

            sgd_update_step_location = self.update_step_size  # already sgd
            sample_interval = self.sample_interval
            resolution = self.resolution
            max_interval_enlargements = self.max_interval_enlargements

            all_measure_locations = []
            all_train_line_losses = []
            all_val_line_losses = []
            all_direction_batch_losses = []

            for remeasure in range(max_interval_enlargements):

                left_measure_locations = list(
                    np.arange(-sample_interval * (remeasure + 1), -sample_interval * remeasure, resolution))
                right_measure_locations = list(
                    np.arange(sample_interval * remeasure, sample_interval * (remeasure + 1), resolution))
                left_measure_locations.extend(right_measure_locations)
                measure_locations = left_measure_locations

                if remeasure == 0:
                    measure_locations.append(sgd_update_step_location)
                    if 0.0 not in measure_locations:
                        measure_locations.append(0.0)
                sorted_measure_locations = list(np.sort(measure_locations))

                all_measure_locations.extend(sorted_measure_locations)
                train_line_losses = np.zeros((train_dataset_size, len(sorted_measure_locations)))
                direction_batch_losses = np.zeros((batch_size, len(sorted_measure_locations)))
                val_line_losses = np.zeros((val_dataset_size, len(sorted_measure_locations)))

                assert (train_dataset_size % batch_size) == 0
                assert (val_dataset_size % batch_size) == 0

                loss_closures_train = []
                loss_closures_val = []

                for i, location in enumerate(sorted_measure_locations):
                    step_size = location - self.current_position
                    self.current_position += step_size
                    self._perform_sgd_update(self.params, step_size)

                    for s in range(train_steps_per_epoch):
                        if i == 0:
                            seed = time.time()
                            (x, y) = next(train_dataset)
                            loss_fn_deterministic = self.get_deterministic_loss_function(x, y, seed)
                            loss_closures_train.append(loss_fn_deterministic)
                            losses = loss_fn_deterministic()

                        else:
                            losses = loss_closures_train[s]()

                        losses = losses.cpu().detach().numpy()

                        loss_index_start = s * batch_size
                        loss_index_end = min((s + 1) * batch_size, train_dataset_size)  # drop last ist active
                        train_line_losses[loss_index_start:loss_index_end, i] = losses

                    _, dir_losses = direction_batch_loss_fn()
                    dir_losses = dir_losses.cpu().numpy()
                    direction_batch_losses[:, i] = dir_losses[:]

                    for s in range(val_steps_per_epoch):
                        with self.random_seed_torch(s):
                            if i == 0:
                                seed = time.time()
                                (x, y) = next(val_dataset)
                                loss_fn_deterministic = self.get_deterministic_loss_function(x, y, seed)

                                loss_closures_val.append(loss_fn_deterministic)
                                losses = loss_fn_deterministic()
                            else:
                                losses = loss_closures_val[s]()
                            losses = losses.cpu().detach().numpy()
                            loss_index_start = s * batch_size
                            loss_index_end = min((s + 1) * batch_size, val_dataset_size)
                            val_line_losses[loss_index_start:loss_index_end, i] = losses
                all_val_line_losses.append(val_line_losses)
                all_train_line_losses.append(train_line_losses)
                all_direction_batch_losses.append(direction_batch_losses)
                mean_val_line_losses = np.mean(val_line_losses, 0)
                min_mean_val_index = mean_val_line_losses.argmin()
                continue_ = False
                if min_mean_val_index == 0 or min_mean_val_index + 1 == len(mean_val_line_losses):
                    continue_ = True
                else:
                    mean_train_line_losses = np.mean(train_line_losses, 0)
                    min_mean_train_index = mean_train_line_losses.argmin()
                    if min_mean_train_index == 0 or min_mean_train_index + 1 == len(mean_train_line_losses):
                        continue_ = True
                    else:
                        min_dir_index = direction_batch_losses.argmin()
                        if min_dir_index == 0 or min_dir_index + 1 == len(direction_batch_losses):
                            continue_ = True

                if continue_ is False:
                    break
                print("resample:", remeasure)

            if remeasure == max_interval_enlargements:
                self.is_extrapolation = True
            else:
                self.is_extrapolation = False

            sort_indexes = np.argsort(all_measure_locations)
            all_measure_locations = np.array(all_measure_locations)[sort_indexes]
            train_line_losses = np.hstack(all_train_line_losses)
            train_line_losses = train_line_losses[:, sort_indexes]
            val_line_losses = np.hstack(all_val_line_losses)
            val_line_losses = val_line_losses[:, sort_indexes]
            direction_batch_losses = np.hstack(all_direction_batch_losses)
            direction_batch_losses = direction_batch_losses[:, sort_indexes]

            path = self.line_data_save_dir + "/line_data_{}.pickle".format(self.performed_training_steps)
            with open(path, 'wb') as f:
                pickle.dump((train_line_losses, val_line_losses, all_measure_locations, direction_batch_losses,
                             self.update_step_size, self.direction_norm, self.is_extrapolation, direction_batch_number),
                            f)
            mean_direction_batch_losses = np.nanmean(direction_batch_losses, axis=0)
            zero_index = np.where(all_measure_locations == 0)[0][0]
            gradients = np.gradient(mean_direction_batch_losses[zero_index - 1:zero_index + 2],
                                    all_measure_locations[zero_index - 1:zero_index + 2])

            print("direction_norm {}  approximated directional derivative {}".format(self.direction_norm, gradients[1]))

            mean_train_loss = np.nanmean(train_line_losses, axis=0)
            mean_val_loss = np.nanmean(val_line_losses, axis=0)
            min_index = np.argmin(mean_train_loss)
            min_mean_train_loss = mean_train_loss[min_index]
            min_mean_train_location = all_measure_locations[min_index]

            loss_sgd_step_index = self._find_nearest(all_measure_locations, sgd_update_step_location)
            loss_sgd_step = mean_train_loss[loss_sgd_step_index]

            near_0_index = self._find_nearest(all_measure_locations, 0.0)  # should be inside but just to be save
            self._log_print("line_number: ", self.performed_training_steps)
            self._log_print("loss_at_0", mean_train_loss[near_0_index])
            self._log_print("loss_min", min_mean_train_loss)
            self._log_print("loss_sgd_step", loss_sgd_step)
            if self.is_plot:
                matplotlib.use('Agg')
                plt.figure()
                plt.plot(all_measure_locations, mean_train_loss, label="train_mean")
                plt.plot(all_measure_locations, mean_val_loss, label="val_mean")
                plt.plot(all_measure_locations, train_line_losses[0, :], label="fist_batch")
                plt.legend()
                dir_ = self.line_data_save_dir
                os.makedirs(dir_, exist_ok=True)
                path = self.line_data_save_dir + "/line_{}.png".format(self.performed_training_steps)
                plt.savefig(path)
                plt.close()
            return min_mean_train_location, min_mean_train_loss

    @staticmethod
    def _find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def _update_direction_vars_and_norm(self, params):
        """
        Update the search or update direction. In SGD training mode it is used as the update direction.
        In measure line mode the normalized direction is used as search direction.
        :param params: the network parameters
        """
        with torch.no_grad():
            norm = torch.tensor(0.0)
            for p in params:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'dir_buffer' not in param_state:
                    buf = param_state['dir_buffer'] = torch.zeros_like(p.grad.data, device=p.device)
                else:
                    buf = param_state['dir_buffer']
                buf.mul_(self.momentum)  # _ = inplace
                buf.add_(p.grad.data)
                flat_buf = buf.view(-1)
                norm = norm + torch.dot(flat_buf, flat_buf)
            torch.sqrt_(norm)
            if norm == 0.0:
                norm = self.epsilon

            if torch.cuda.is_available() and isinstance(norm, torch.Tensor):
                self.direction_norm = norm.cuda()
            else:
                self.direction_norm = norm

    def _set_checkpoint(self):
        """
        Saves the current position on the parameter space
        """
        with torch.no_grad():
            self.model_state_checkpoint = copy.deepcopy(self.model.state_dict())
            for p in self.params:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state['ckpt_buffer'] = param_state['dir_buffer'].clone().detach()

    def _reset_to_best_checkpoint(self):
        """
        Resets to the parameter position to the last saved checkpoint
        """
        with torch.no_grad():
            if self.model_state_checkpoint is not None:
                self.model.load_state_dict(self.model_state_checkpoint)
                self.params = list(self.model.parameters())
                for p in self.params:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    assert "ckpt_buffer" in param_state
                    param_state['dir_buffer'] = param_state['ckpt_buffer'].clone().detach()

    def _perform_sgd_update(self, params, step_size):
        """
        Performs a SGD update step.
        """
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                mom = self.state[p]["dir_buffer"]
                p.data += step_size * -mom / self.direction_norm

    def _get_l2_loss(self, params):
        """
        Determines the current l2 loss.
        """
        l2_loss = torch.tensor(0.0)
        if self.l2_reg is not 0.0:
            for p in params:
                if p.data is None:
                    continue
                flat_data = p.data.view(-1)
                l2_loss = l2_loss + torch.dot(flat_data, flat_data)
        return l2_loss * self.l2_reg * 0.5

    def _perform_update_step(self, params, step_size, loss_fn, new_direction):
        """
        Performs an update step on the parameters space in the direction of the direction variables.
        :param new_direction: if true, update the direction variables
        :return: current loss, model output
        """
        with torch.no_grad():
            if self.fist_update_call:
                with torch.enable_grad():
                    loss = loss_fn()  # at the first step we have to call the loss_fn twice to initialize the direction buffer
                    l2_loss = self._get_l2_loss(params)
                    loss = loss + l2_loss
                    loss.backward()
                    with torch.no_grad():
                        self._update_direction_vars_and_norm(params)
                        self._set_checkpoint()
                    self.fist_update_call = False

            self._perform_sgd_update(params, step_size)

            if new_direction:
                with torch.enable_grad():
                    loss = loss_fn()
                    l2_loss = self._get_l2_loss(params)
                    loss = loss + l2_loss
                    loss.backward()
                    loss = self._to_numpy(loss)
            else:
                loss = loss_fn()
                l2_loss = self._get_l2_loss(params)
                loss = loss + l2_loss
                loss = self._to_numpy(loss)

            self.current_position += step_size
            if new_direction:
                self._update_direction_vars_and_norm(params)
                self.current_position = 0
            return loss

    def step(self, loss_fn):
        """
        # No support of param groups since it conflicts with checkpoints of the parameters and every state field must be param_group specific
        :param loss_fn: function of the form:
         >>>   def closure():
         >>>       self.optimizer.zero_grad()
         >>>       output = self.model(x)
         >>>       losses = self.loss2(output, y)
         >>>       loss_ =  torch.mean(losses)
         >>>       output_placeholder.append(output)
         >>>       return loss_,losses
        :return: current loss, current model output, current step size if SGD training state. None,None,None if Line Search State.
        """

        def loss_function_deterministic():
            with self.random_seed_torch(int(self.performed_training_steps)):
                l = loss_fn()
            return l

        def loss_fn_normal():
            loss, losses = loss_function_deterministic()
            return loss

        with torch.no_grad():
            self.performed_training_steps += 1
            self.zero_grad()
            if len(self.lr_milestones) is not 0:
                if self.performed_training_steps >= self.lr_milestones[0]:
                    del self.lr_milestones[0]
                    self.sgd_lr *= 0.1

            params = self.params

            self._perform_update_step(params, 0, loss_fn_normal, new_direction=True)
            self.update_step_size = (self.sgd_lr * self.direction_norm).item()
            _, _ = self.measure_exact_loss(loss_function_deterministic, self.performed_training_steps)
            step_size = self.update_step_size - self.current_position
            batch_loss = self._perform_update_step(params, step_size, loss_fn_normal, new_direction=False)

            if (self.performed_training_steps % 100) == 0:
                self.save_checkpoint(self.performed_training_steps, self.line_data_save_dir),

            return batch_loss, self.sgd_lr

    def save_checkpoint(self, step, directory):
        path = os.path.join(directory, "step_{0}.cpt".format(step))
        with open(path, 'wb') as f:
            torch.save({
                'step': step,
                'model_state_dict': self.model.state_dict(),
            }, f)
        self._log_print("saved checkpoint at step: {0}".format(step))

    def _log_print(self, *messages):
        if self.is_logging: print(*messages)

    @staticmethod
    def _get_mean_least_square_error(x, y, fitted_function):
        assert len(x) == len(y)
        assert len(x) > 0
        return np.sum((fitted_function(x) - y) ** 2) / len(x)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().detach().numpy()
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                return [x.cpu().detach().numpy() for x in value]
        return value

    @staticmethod
    @contextlib.contextmanager
    def random_seed_torch(seed, device=0):
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state(0)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state, device)

    def state_dict(self):
        """
        :return:  a dictionary representing the current state of the optimizer
        """
        # dict_ = self.__dict__
        # return copy.deepcopy(dict_)

        return {}

    def load_state_dict(self, state_dict):
        """
        set the current state of the optimizer
        """
        state_dict = copy.deepcopy(state_dict)
        self.__dict__.update(state_dict)
        self.params = list(self.model.parameters())
