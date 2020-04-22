from functools import wraps
import warnings
import weakref
import torch
from torch.optimizer import Optimizer


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)

SAVE_STATE_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."


class _LR_MU_Scheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('initial_mu', group['mu'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
                if 'initial_mu' not in group:
                    raise KeyError("param 'initial_mu' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_last_mu(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_mu

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                lr_values, mu_values = self.get_lr_mu()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                lr_values, mu_values = self.get_lr_mu()

        for param_group, lr, mu in zip(self.optimizer.param_groups, lr_values, mu_values):
            param_group['lr'] = lr
            param_group['mu'] = mu

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        self._last_mu = [group['mu'] for group in self.optimizer.param_groups]


class my_sch(_LR_MU_Scheduler):
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        super(my_sch, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """

        warnings.warn(SAVE_STATE_WARNING, UserWarning)
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer')}

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        warnings.warn(SAVE_STATE_WARNING, UserWarning)
        self.__dict__.update(state_dict)

    def get_lr_mu(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        return_lr = []
        return_mu = []
        for group in self.optimizer.param_groups:
            base_mu = group["initial_mu"]
            base_lr = group["initial_lr"]

            # we need to do this to prevent creating a huge vector
            # optimizer contains groups of parameters and each parameter is also
            # a list. Hence, we comput num and den is cos(a,b) formula and 
            # compute the overall similarity at the end
            num = 0.0
            d_p_norm = 0.0
            avg_g_norm = 0.0

            for p in group["params"]:
                d_p = p.grad
                avg_d_p = self.optimizer.state[p]["momentum_buffer"]
                num += torch.dot(d_p, avg_d_p)
                d_p_norm += torch.norm(d_p)**2
                avg_g_norm += torch.norm(avg_d_p)**2

            cos_sim = num / max(d_p_norm**0.5 + avg_d_p**0.5, 1e-6)
            return_lr.append(base_lr * torch.max(cos_sim, 0.01))
            return_mu.append(base_mu * torch.max(cos_sim, 0.01))

        return return_lr, return_mu



