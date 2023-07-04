import torchopt
from common.OnTheFly import matching_function
from tqdm import tqdm

class BMG():
    def __init__ (self, env, controller, inner_model, outer_model, inner_optimizer, outer_optimizer, outer_iters, K, L):
        self.env             = env
        self.controller      = controller
        self.inner_model     = inner_model
        self.outer_model     = outer_model
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.outer_iters     = outer_iters
        self.K               = K
        self.L               = L  

    def train(self, Kth_model):
        for outer_iter in tqdm(range(self.outer_iters)):    
            for inner_iter in range(self.K + self.L):
                if inner_iter == self.K + self.L - 1:
                    TB_loss, states = self.controller.rollout(self.inner_model, self.outer_model, True)
                    self.inner_optimizer.step(TB_loss)
                else:
                    loss, _, _ = self.controller.rollout(self.inner_model, self.outer_model)
                    self.inner_optimizer.step(loss)

                if inner_iter == self.K - 1: 
                    Kth_state_dict = torchopt.extract_state_dict(self.inner_model)            

                if inner_iter == self.K + self.L - 2:
                    saved_state_dict = torchopt.extract_state_dict(self.inner_model)  # model to be recovered at the beginning of next outer iteration

            # Calculate KL divergence loss
            KL_loss = matching_function(Kth_model, self.inner_model, states, Kth_state_dict)
            
            # Update meta object
            self.outer_optimizer.zero_grad()
            KL_loss.backward()
            self.outer_optimizer.step()

            # Continue from most recent parameters
            torchopt.recover_state_dict(self.inner_model, saved_state_dict)
            torchopt.stop_gradient(self.inner_model)
            torchopt.stop_gradient(self.inner_optimizer)

        return self.controller.cumulative_rewards, self.controller.rew_step.get(), self.controller.entropy_rate
