import torch
import numpy as np

from torch.nn.utils.clip_grad import clip_grad_norm_
from maml.utils import accuracy,shannon_entropy

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)

    return total_norm

class MetaLearner(object):
    def __init__(self, model, embedding_model, optimizers, fast_lr, loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 collect_accuracies, device, alternating=False,
                 embedding_schedule=10, classifier_schedule=10,
                 embedding_grad_clip=0, entropy_alpha = False,
                 DAnet=None, Loss_list=None, dsn_num = False, dsnNetNum = False):
        self._model = model
        self._embedding_model = embedding_model
        self._fast_lr = fast_lr
        self._optimizers = optimizers
        self._loss_func = loss_func
        self._first_order = first_order
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._collect_accuracies = collect_accuracies
        self._device = device
        self._alternating = alternating
        self._alternating_schedules = (classifier_schedule, embedding_schedule)
        self._alternating_count = 0
        self._alternating_index = 1
        self._embedding_grad_clip = embedding_grad_clip
        self._grads_mean = []
        self._entropy_alpha = entropy_alpha
        self._entropy_mean = []
        self._entropyH2_mean = []
        self._entropyH1_mean = []
        self._da_net = DAnet
        self._loss_list = Loss_list
        self._dsn_num = dsn_num
        self._dsnNetNum = dsnNetNum
        self.to(device)

        self._reset_measurements()

    def _reset_measurements(self):
        self._count_iters = 0.0
        self._cum_loss = 0.0
        self._cum_accuracy = 0.0

    def _update_measurements(self, task, loss, preds):
        self._count_iters += 1.0
        self._cum_loss += loss.data.cpu().numpy()
        if self._collect_accuracies:
            self._cum_accuracy += accuracy(
                preds, task.y).data.cpu().numpy()

    def _pop_measurements(self):
        measurements = {}
        loss = self._cum_loss / self._count_iters
        measurements['loss'] = loss
        if self._collect_accuracies:
            accuracy = self._cum_accuracy / self._count_iters
            measurements['accuracy'] = accuracy
        self._reset_measurements()
        return measurements

    def measure(self, tasks, train_tasks=None, adapted_params_list=None,
                embeddings_list=None):
        """Measures performance on tasks. Either train_tasks has to be a list
        of training task for computing embeddings, or adapted_params_list and
        embeddings_list have to contain adapted_params and embeddings"""
        if adapted_params_list is None:
            adapted_params_list = [None] * len(tasks)
        if embeddings_list is None:
            embeddings_list = [None] * len(tasks)
        for i in range(len(tasks)):
            params = adapted_params_list[i]
            if params is None:
                params = self._model.param_dict
            embeddings = embeddings_list[i]
            task = tasks[i]
            preds = self._model(task, params=params, embeddings=embeddings)
            loss = self._loss_func(preds, task.y)
            self._update_measurements(task, loss, preds)

        measurements = self._pop_measurements()
        return measurements

    def measure_each(self, tasks, train_tasks=None, adapted_params_list=None,
                     embeddings_list=None):
        """Measures performance on tasks. Either train_tasks has to be a list
        of training task for computing embeddings, or adapted_params_list and
        embeddings_list have to contain adapted_params and embeddings"""
        """Return a list of losses and accuracies"""
        if adapted_params_list is None:
            adapted_params_list = [None] * len(tasks)
        if embeddings_list is None:
            embeddings_list = [None] * len(tasks)
        accuracies = []
        for i in range(len(tasks)):
            params = adapted_params_list[i]
            if params is None:
                params = self._model.param_dict
            embeddings = embeddings_list[i]
            task = tasks[i]
            preds = self._model(task, params=params, embeddings=embeddings)
            pred_y = np.argmax(preds.data.cpu().numpy(), axis=-1)
            accuracy = np.mean(
                task.y.data.cpu().numpy() == 
                np.argmax(preds.data.cpu().numpy(), axis=-1))
            accuracies.append(accuracy)

        return accuracies

    def update_params(self, loss, params):
        """Apply one step of gradient descent on the loss function `loss`,
        with step-size `self._fast_lr`, and returns the updated parameters.
        """
        create_graph = not self._first_order
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if self._inner_loop_grad_clip > 0 and grad is not None:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                                  max=self._inner_loop_grad_clip)
            if grad is not None:
              params[name] = param - self._fast_lr * grad

        return params

    def adapt(self, train_tasks):
        adapted_params = []
        embeddings_list = []
        dsn_losses = []
        flag = 0
        for task in train_tasks:
            params = self._model.param_dict
            embeddings = None
            if self._embedding_model:
                embeddings, embedding_input = self._embedding_model(task)
                task_st = task.task_st
                # DSN NET
                if self._dsn_num:
                    DSNLoss = 0
                    alpha_weight = self._dsn_num
                    beta_weight = self._dsn_num
                    newembeddings = []
                    if task_st == 'source':
                        target_privte_code, target_share_code, _, source_class_label, target_rec_code, embedding128, embedding512 = self._da_net(
                            input_data=embedding_input,
                            mode='source',
                            rec_scheme='all')
                    else:
                        target_privte_code, target_share_code, _ , target_rec_code, embedding128, embedding512 = self._da_net(input_data=embedding_input,
                                                                                                 mode='target',
                                                                                                 rec_scheme='all')#
                    theWholeList = [embedding512,target_privte_code, embedding128, target_share_code ]
                    if self._dsnNetNum == 4:
                        for i in range(self._dsnNetNum):
                            newembeddings.append(theWholeList[3-i])
                    else:
                        for i in range(self._dsnNetNum):
                            newembeddings.append(theWholeList[i])
                    embeddings = newembeddings
                    target_diff = beta_weight * self._loss_list[3](target_privte_code, target_share_code)
                    DSNLoss += target_diff
                    target_mse = alpha_weight * self._loss_list[1](target_rec_code, embedding_input)
                    DSNLoss += target_mse
                    target_simse = alpha_weight * self._loss_list[2](target_rec_code, embedding_input)
                    DSNLoss += target_simse
                    dsn_losses.append(DSNLoss)

            for i in range(self._num_updates):
                preds = self._model(task, params=params, embeddings=embeddings)
                loss = self._loss_func(preds, task.y)
                params = self.update_params(loss, params=params)
                if i == 0:
                    self._update_measurements(task, loss, preds)
            adapted_params.append(params)
            embeddings_list.append(embeddings)
        if len(dsn_losses) != 0 :
            dsnLoss = torch.mean(torch.stack(dsn_losses))
        else:
            dsnLoss = None
        measurements = self._pop_measurements()
        return measurements, adapted_params, embeddings_list, dsnLoss

    def step(self, adapted_params_list, embeddings_list, val_tasks,
             is_training,DSNLoss):
        if self._dsn_num:
            self._optimizers[2].zero_grad()
            DSNLoss.backward(retain_graph=True)
            self._optimizers[2].step()

        for optimizer in self._optimizers:
            optimizer.zero_grad()
        post_update_losses = []
        for adapted_params, embeddings, task in zip(
                adapted_params_list, embeddings_list, val_tasks):


            if self._entropy_alpha:
                # with entropy
                preds_before = self._model(task, params=self._model.param_dict,
                                           embeddings=embeddings)

                preds = self._model(task, params=adapted_params,
                                    embeddings=embeddings)
                loss = self._loss_func(preds, task.y)


                entropyloss = shannon_entropy(preds) - shannon_entropy(preds_before)
                self._entropy_mean.append(float(entropyloss))
                self._entropyH2_mean.append(float(shannon_entropy(preds)))
                self._entropyH1_mean.append(float(shannon_entropy(preds_before)))

                loss += self._entropy_alpha * entropyloss

            else:
                # without entropy

                preds = self._model(task, params=adapted_params,
                                    embeddings=embeddings)
                loss = self._loss_func(preds, task.y)

            post_update_losses.append(loss)
            self._update_measurements(task, loss, preds)

        mean_loss = torch.mean(torch.stack(post_update_losses))
        if is_training:
            mean_loss.backward()
            if self._alternating:
                self._optimizers[self._alternating_index].step()
                self._alternating_count += 1
                if self._alternating_count % self._alternating_schedules[self._alternating_index] == 0:
                    self._alternating_index = (1 - self._alternating_index)
                    self._alternating_count = 0
            else:
              self._optimizers[0].step()
              if len(self._optimizers) > 1:
                  if self._embedding_grad_clip > 0:
                      _grad_norm = clip_grad_norm_(self._embedding_model.parameters(), self._embedding_grad_clip)
                  else:
                      _grad_norm = get_grad_norm(self._embedding_model.parameters())

                  self._grads_mean.append(_grad_norm)
                  self._optimizers[1].step()
                  if self._dsn_num:
                    self._optimizers[2].step()

        measurements = self._pop_measurements()
        return measurements

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)
        if self._embedding_model:
            self._embedding_model.to(device, **kwargs)

    def state_dict(self):
        state = {
            'model_state_dict': self._model.state_dict(),
            'optimizers': [ optimizer.state_dict() for optimizer in self._optimizers ]
        }
        if self._embedding_model:
            state.update(
                {'embedding_model_state_dict':
                    self._embedding_model.state_dict()})
        return state
