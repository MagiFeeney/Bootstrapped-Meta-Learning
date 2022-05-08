import TorchOpt
import meta_optim as MetaOptim
from tqdm import tqdm

class Trainer():
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

    def train(self, model, opt=None, flag="TorchOpt"):
        if flag == "TorchOpt":
            cumulative_rewards, rew_step, entropy_rate = self.BMGTorchOpt(model)
        elif flag == "MetaOptim":
            cumulative_rewards, rew_step, entropy_rate = self.BMGMetaOptim(model, opt)
            
        return cumulative_rewards, rew_step, entropy_rate

    def BMGTorchOpt(self, Kth_model):
        for outer_iter in tqdm(range(self.outer_iters)):    
            for inner_iter in range(self.K + self.L):
                if inner_iter == self.K + self.L - 1:
                    TB_loss, states = self.controller.rollout(self.inner_model, self.outer_model, TB_flag=True)
                    self.inner_optimizer.step(TB_loss)
                else:
                    loss = self.controller.rollout(self.inner_model, self.outer_model)
                    self.inner_optimizer.step(loss)

                if inner_iter == self.K - 1: 
                    Kth_state_dict = TorchOpt.extract_state_dict(self.inner_model)            

                if inner_iter == self.K + self.L - 2:
                    saved_state_dict = TorchOpt.extract_state_dict(self.inner_model)  # model to be recovered at the beginning of next outer iteration

            # Calculate KL divergence loss
            KL_loss = self.controller.matching_function(Kth_model, self.inner_model, states, Kth_state_dict)
            
            # Update meta object
            self.outer_optimizer.zero_grad()
            KL_loss.backward()
            self.outer_optimizer.step()

            # Continue from most recent parameters
            TorchOpt.recover_state_dict(self.inner_model, saved_state_dict)
            TorchOpt.stop_gradient(self.inner_model)
            TorchOpt.stop_gradient(self.inner_optimizer)

        return self.controller.cumulative_rewards, self.controller.rew_step.get(), self.controller.entropy_rate

    def BMGMetaOptim(self, bootstrap_model, bootstrap_optimizer):
        for outer_iter in tqdm(range(self.outer_iters)):    
            with MetaOptim.SlidingWindow(offline=False):
                for inner_iter in range(self.K + self.L):

                    if inner_iter < self.K:
                        loss = self.controller.rollout(self.inner_model, self.outer_model)
                        self.inner_optimizer.step(loss)
                        if inner_iter == self.K - 1: 
                            # copy inner model for not intervening parameters of meta obeject anymore   
                            bootstrap_model.load_state_dict(self.inner_model.state_dict())    
                    else:
                        if inner_iter < self.K + self.L - 1:  
                            loss = self.controller.rollout(bootstrap_model, self.outer_model)
                            bootstrap_optimizer.zero_grad()
                            loss.backward()
                            bootstrap_optimizer.step()
                            if inner_iter == self.K + self.L - 2:
                                saved_model = bootstrap_model.state_dict() # save model at (K + L - 1)th rollout
                        elif inner_iter == self.K + self.L - 1:
                            TB_loss, states = self.controller.rollout(bootstrap_model, self.outer_model, TB_flag=True)
                            bootstrap_optimizer.zero_grad()
                            TB_loss.backward()
                            bootstrap_optimizer.step()

                # Calculate KL divergence loss
                KL_loss = self.controller.matching_function(self.inner_model, bootstrap_model, states, flag=False)
                
                # Update meta object
                self.outer_optimizer.zero_grad()
                KL_loss.backward()
                self.outer_optimizer.step()

                # Continue from most recent parameters
                self.inner_model.load_state_dict(saved_model)

        return self.controller.cumulative_rewards, self.controller.rew_step.get(), self.controller.entropy_rate