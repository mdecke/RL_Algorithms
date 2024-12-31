import numpy as np
import optuna


class EM_Algorithm:
    def __init__(self, data, n_distributions, regularization_constant, max_iterations=1000, tol=1e-8):
        self.data = data
        self.n_dist = n_distributions
        self.lambda_reg = regularization_constant
        self.max_it = max_iterations
        self.tol = tol
        self._initialize_parameters()

    def _initialize_parameters(self):
        self.means = np.linspace(min(self.data), max(self.data), self.n_dist)
        self.stds = np.full(self.n_dist, np.std(self.data))
        self.weights = np.full(self.n_dist, 1 / self.n_dist)

    def gaussian(self, x, mu, std):
        if std == 0:
            std = 1e-6
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)
    
    def _compute_log_likelihood(self, regularized=False):
        log_likelihood = np.sum(np.log(np.sum([
            self.weights[k] * self.gaussian(self.data, self.means[k], self.stds[k]) 
            for k in range(self.n_dist)
        ], axis=0)))
        
        if regularized:
            log_likelihood -= self.lambda_reg * np.sum(1 / self.stds)
        return log_likelihood

    def fit(self, regularized=True):
        n = len(self.data)
        posteriors = np.zeros((n, self.n_dist))
        prev_log_likelihood = -np.inf

        for _ in range(self.max_it):
            # E-step
            for k in range(self.n_dist):
                posteriors[:, k] = self.weights[k] * self.gaussian(self.data, self.means[k], self.stds[k])
            posteriors /= posteriors.sum(axis=1, keepdims=True)

            # M-step
            for k in range(self.n_dist):
                N_k = posteriors[:, k].sum()
                self.means[k] = (posteriors[:, k] @ self.data) / N_k
                std_numerator = posteriors[:, k] @ ((self.data - self.means[k])**2)
                self.stds[k] = np.sqrt(std_numerator / N_k + (self.lambda_reg / (self.stds[k]+1e-8)**2 if regularized else 0))
                self.weights[k] = N_k / n
            
            log_likelihood = self._compute_log_likelihood(regularized)
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        component_indices = np.random.choice(
            self.n_dist, 
            size=n_samples, 
            p=self.weights
        )
        
        samples = np.zeros(n_samples)
        for i in range(n_samples):
            idx = component_indices[i]
            samples[i] = np.random.normal(
                loc=self.means[idx],
                scale=self.stds[idx]
            )
            
        return samples

class EMOptunaOptimizer:
    def __init__(self, data, n_trials=100, n_components_range=(2, 10), lambda_reg_range=(1e-6, 1e-2)):
        self.data = data
        self.n_trials = n_trials
        self.n_components_range = n_components_range
        self.lambda_reg_range = lambda_reg_range
        self.study = None

    def _objective(self, trial):
        n_components = trial.suggest_int("n_components", *self.n_components_range)
        lambda_reg = trial.suggest_float("lambda_reg", *self.lambda_reg_range, log=True)

        em = EM_Algorithm(self.data, n_components, lambda_reg)
        em.fit(regularized=True)
        return -em._compute_log_likelihood(regularized=True)

    def optimize(self):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self._objective, n_trials=self.n_trials)

    def get_best_params(self):
        if self.study is None:
            raise ValueError("Run optimize() first")
        return self.study.best_params

    def print_best_results(self):
        if self.study is None:
            raise ValueError("Run optimize() first")
        print(f"Best trial:")
        print(f"  n_components: {self.study.best_params['n_components']}")
        print(f"  lambda_reg: {self.study.best_params['lambda_reg']}")
        print(f"  Best value (negative log-likelihood): {self.study.best_value}")
