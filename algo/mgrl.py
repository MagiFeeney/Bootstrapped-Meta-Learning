import torchopt
from tqdm import tqdm

class MGRL():
    def __init__ (self, env, controller, inner_model, outer_model, inner_optimizer, outer_optimizer, outer_iters, epsilon_meta, T):
        self.env             = env
        self.controller      = controller
        self.inner_model     = inner_model
        self.outer_model     = outer_model
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.outer_iters     = outer_iters
        self.epsilon_meta    = epsilon_meta
        self.T               = T

    def train(self):
        for outer_iter in tqdm(range(self.outer_iters)):    
            meta_loss = 0
            for inner_iter in range(self.T):
                loss, actor_loss, entropy_loss = self.controller.rollout(self.inner_model, self.outer_model)
                self.inner_optimizer.step(loss)
                meta_loss = meta_loss + ((actor_loss + self.epsilon_meta * entropy_loss) - meta_loss) / (inner_iter + 1)

            # Update meta object
            self.outer_optimizer.zero_grad()
            meta_loss.backward()
            self.outer_optimizer.step()

            # Stop gradient to start next outer update
            torchopt.stop_gradient(self.inner_model)
            torchopt.stop_gradient(self.inner_optimizer)

        return self.controller.cumulative_rewards, self.controller.rew_step.get(), self.controller.entropy_rate

