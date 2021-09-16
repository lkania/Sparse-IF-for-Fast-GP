from logger import Logger
import tensorflow as tf
import gpflow

class GP():

    def __init__(self,
                 variance_init,
                 lenghscale_init,
                 gaussian_noise_init,
                 float32=False,
                 seed=123):

        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(seed)

        if float32:
            self.dtype = tf.float32
        else:
            self.dtype = tf.float64

        gpflow.config.set_default_float(self.dtype)

        self.D = len(lenghscale_init)

        self.gaussian_noise_init = tf.cast(gaussian_noise_init,dtype=self.dtype)
        self.kernel=gpflow.kernels.RBF(
            lengthscales=tf.cast(lenghscale_init,dtype=self.dtype),
            variance=tf.cast(variance_init,dtype=self.dtype))

    @property
    def lengthscales(self):
        return self.model.kernel.lengthscales.numpy()

    @property
    def inv_lengthscales(self):
        return (1/self.model.kernel.lengthscales).numpy()

    @property
    def variance(self):
        return self.model.kernel.variance.numpy()

    @property
    def gaussian_noise(self):
        return self.model.likelihood.variance.numpy()

    def _predict(self,Xs,full_cov=False):
        mean,covar = self.model.predict_y(Xnew=Xs,full_cov=full_cov)

        return mean,covar

    def optimize(self,
                 iterations,
                 Xtrain, Ytrain, Xval, Yval,
                 path,
                 run=0,
                 Ystd=None):

        Xtrain = tf.convert_to_tensor(Xtrain, dtype=self.dtype)
        Ytrain = tf.convert_to_tensor(Ytrain, dtype=self.dtype)
        Xval = tf.convert_to_tensor(Xval, dtype=self.dtype)
        Yval = tf.convert_to_tensor(Yval, dtype=self.dtype)

        self.model = gpflow.models.GPR((Xtrain, Ytrain), kernel=self.kernel)
        self.model.likelihood.variance.assign(self.gaussian_noise_init)

        logger = Logger(log_dir='{0}GP/run_{1}'.format(path, run), model=self,
                        Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval,
                        Ystd=Ystd)

        logger.collect(it=0, data_points_used=0, record=True)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(closure=self.model.training_loss_closure(),
                     variables=self.model.trainable_variables,
                     options=dict(disp=True,maxiter=iterations))

        logger.collect(it=1, data_points_used=Xtrain.shape[0], record=True)