# improved stochastic optimal; transport plan with interpolation
import numpy as np
import ot as pot

from metric import of_uni_cate, similarity_to_dissimilarity
from multiprocessing import Pool


def multicore_helper(args):
    func, target, repeat, interpolation = args
    return func.transform(target, repeat, interpolation)


class adaptation():
    def __init__(self, cate_dim, num_dim):
        self.cate_dim = cate_dim
        self.num_dim = num_dim

    def fit(self, target, source, lmbda=None, **kwargs):
        self.adapt = []

        for i in range(self.cate_dim + self.num_dim):
            if i < self.cate_dim:
                adapt = adaptation_cate_1d()
                adapt.fit(target[:,i], source[:,i], lmbda=lmbda, **kwargs)
            else:
                adapt = adaptation_num_1d()
                adapt.fit(target[:,i], source[:,i], lmbda=lmbda, **kwargs)
            self.adapt.append(adapt)

    def transform(self, target, repeat=1, interpolation=1, njobs=1):
        args = []
        
        if isinstance(interpolation, int) or isinstance(interpolation, float):
            interpolation = [interpolation] * (self.cate_dim + self.num_dim)

        for i in range(self.cate_dim + self.num_dim):
            args.append([self.adapt[i], target[:,i].copy(), repeat, interpolation[i]])
        
        with Pool(njobs) as p:
            target_trans = p.map(multicore_helper, args)
        
        return np.array(target_trans).T


class adaptation_cate_1d():
    def fit(self, target, source, distance=None, lmbda=None, **args):
        self.identical = False
        if len(target)==0 or len(source)==0:
            self.identical = True
        else:
            # Get similarity matrix
            if distance is None:
                sim, modality = of_uni_cate(source, target)
                distance = similarity_to_dissimilarity(sim)
            else:
                modality = np.unique(np.r_[target, source])
            
            # Get target modality
            self.target_modality, counts = np.unique(target, return_counts=True)
            self.target_density = counts / counts.sum()

            # Get source modality
            self.source_modality, counts = np.unique(source, return_counts=True)
            self.source_density = counts / counts.sum()
            
            # Compute transportation plan
            target_index = np.where(np.in1d(modality, self.target_modality))[0]
            source_index = np.where(np.in1d(modality, self.source_modality))[0]

            if lmbda is None:
                Gs = pot.emd(self.target_density.tolist(), self.source_density.tolist(), 
                            distance[target_index][:,source_index].tolist())
            else:
                Gs = pot.sinkhorn(self.target_density.tolist(), self.source_density.tolist(), 
                            distance[target_index][:,source_index].tolist(), lmbda)
            
            norm_array = Gs.sum(axis=1)
            norm_array[norm_array==0] = 1
            self.ot_plan = (Gs.T / norm_array).T
            
            # Get stochastic transportation plan in a dict
            self.stochastic_trans = {}
            for m in self.target_modality:
                index = np.where(self.target_modality==m)[0][0]
                self.stochastic_trans[m] = self.ot_plan[index]

    def transform(self, target, repeat=1, interpolation=1, **args):
        if self.identical:
            return np.tile(target, repeat)
        # Transform
        trans_target = np.tile(target, repeat)

        if interpolation > 0.5:
            trans_target_ref = trans_target.copy()
            for m in self.target_modality:
                if np.sum(trans_target_ref==m) > 0:
                    trans_target[trans_target_ref==m] = self.source_modality[np.random.choice(
                        range(len(self.source_modality)), 
                        size=np.sum(trans_target_ref==m), 
                        p=self.stochastic_trans[m])]
        
        return trans_target


class adaptation_num_1d():
    def fit(self, target, source, limit_modality=1000, lmbda=None, **args):
        self.modality = np.unique(np.r_[target, source])
        # density estimation of numerical features
        if len(self.modality) > limit_modality:
            target_source = np.sort(np.r_[target, source])
            step_n = limit_modality
            step = int(len(target_source) / step_n)
            modality_simple = np.unique(target_source[::step])
            while len(modality_simple) < int(limit_modality/2): 
                step_n = step_n * 2
                step = max(int(len(target_source) / step_n), 1)
                modality_simple = np.unique(target_source[::step])

            self.modality = np.unique(np.r_[[self.modality[0]], modality_simple, [self.modality[-1]]])
            
            # take the lower bound to digitize target and source data
            modality_index = np.digitize(target, self.modality) - 1
            target = self.modality[modality_index]
            
            modality_index = np.digitize(source, self.modality) - 1
            source = self.modality[modality_index]
            
        # Compute the distance
        distance = pot.dist(self.modality.reshape(-1, 1), self.modality.reshape(-1, 1))

        self.adapt = adaptation_cate_1d()
        self.adapt.fit(target, source, distance, lmbda)


    def transform(self, target, repeat=1, interpolation=1, **args):

        target = target.copy()
        target[target < self.modality[0]] = self.modality[0]
        
        modality_index = np.digitize(target, self.modality) - 1
        target = self.modality[modality_index]
        
        target_trans = self.adapt.transform(target, repeat)
        return (target_trans - np.tile(target, repeat)) * interpolation + np.tile(target, repeat)
