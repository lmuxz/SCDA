import numpy as np
from sklearn.preprocessing import StandardScaler

class CORAL():
    def __init__(self):
        pass

    def fit(self, source, target, fweights=None):
        self.scaler_source, self.scaler_target = StandardScaler(), StandardScaler()
        source = self.scaler_source.fit_transform(source)
        target = self.scaler_target.fit_transform(target)

        Ct = np.cov(target, rowvar=False, fweights=fweights)# + 1 * np.identity(target.shape[1])
        Cs = np.cov(source, rowvar=False)# + 1 * np.identity(source.shape[1])

        Us, s, _ = np.linalg.svd(Cs)
        Ut, t, _ = np.linalg.svd(Ct)

        s = np.diag(s)
        t = np.diag(t)

        inv_s = np.sqrt(np.linalg.pinv(s))
        t = np.sqrt(t)

        self.whitening = Us.dot(inv_s).dot(Us.T)
        self.coloring = Ut.dot(t).dot(Ut.T)
    
    def transform(self, x):
        x = self.scaler_source.fit_transform(x)
        x_tsf = x.dot(self.whitening).dot(self.coloring)
        return self.scaler_target.inverse_transform(x_tsf)
