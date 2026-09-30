"""Reuse an official OSGPR training graph while resetting Adam at every update."""


def optimizer_entry(model, learning_rate, cache):
    import tensorflow as tf
    key=(tuple(model.X.shape),tuple(model.Y.shape),tuple(model.inducing_variable.Z.shape),
         str(model.X.dtype),float(learning_rate))
    entry=cache.get(key)
    if entry is None:
        # Fixed-shape input variables allow the same official loss graph to read
        # a new legal observation block without tracing another Python model.
        model.X=tf.Variable(model.X,trainable=False)
        model.Y=tf.Variable(model.Y,trainable=False)
        model.data=(model.X,model.Y)
        optimizer=tf.optimizers.Adam(float(learning_rate),jit_compile=False)
        optimizer.build(model.trainable_variables)
        initial=[v.numpy().copy() for v in optimizer.variables()]

        @tf.function(autograph=False,jit_compile=False)
        def step():
            with tf.GradientTape() as tape:
                loss=tf.debugging.check_numerics(model.training_loss(),'Nonfinite Bui objective')
            variables=model.trainable_variables
            gradients=tape.gradient(loss,variables)
            if any(g is None for g in gradients):
                raise FloatingPointError('Missing Bui objective gradient')
            gradients=[tf.debugging.check_numerics(g,'Nonfinite Bui gradient') for g in gradients]
            optimizer.apply_gradients(zip(gradients,variables))
            return loss

        entry=(model,optimizer,initial,step)
        cache[key]=entry
    cached,optimizer,initial,step=entry
    if cached is not model:
        cached.X.assign(model.X);cached.Y.assign(model.Y)
        for name in ['mu_old','Su_old','Kaa_old','Z_old']:
            getattr(cached,name).assign(getattr(model,name))
        if len(cached.trainable_variables)!=len(model.trainable_variables):
            raise ValueError('Reusable OSGPR graph parameter structure changed')
        for destination,source in zip(cached.trainable_variables,model.trainable_variables):
            destination.assign(source)
    for variable,value in zip(optimizer.variables(),initial):
        variable.assign(value)
    return cached,step
