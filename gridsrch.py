from itertools import product
#import core.modules as cm
#from core.metrics import MEE, RMSE
#from core.callbacks import EarlyStopping

def cross_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

#qua andrebbero inseriti i valori su cui fare la search
param_of_search = {
    'nr_hidden_layers' : [1],
    #gli hidden layer dovrebbero poter essere di dimensioni diverse. Dobbiamo provare?
    'size_hidden_layers' : [1],
    'activation_function' : [1],
    'learning_rate' : [1],
    'decay' : [1],
    'momentum' : [1],
    'lambda_reg' : [1],
    'inf_rand' : [1],
    'sup_rand' : [1],
    'patience' : [1],
    'max_epoch' : [1],
    'size_minibatch' : [1]
}

hyperpar_comb = cross_product(param_of_search)

for comb in hyperpar_comb:
    layers = [cm.Input()]
    #questo for va aggiustato. servono le dimensioni di input e output e gli hidden layers possono avere dimensioni diverse
    for _ in range(0, comb['nr_hidden_layers']):
        layers.extend(cm.Dense(
            comb['size_hidden_layers'], 
            comb['size_hidden_layers'], 
            comb['activation_function'],
            #weights_initializer : qualcosa con (comb['inf_rand'], comb['sup_rand']),
            #weights_regularizer : qualcosa con comb['lambda_reg']
        ))
    layers.extend(cm.Linear(comb['size_hidden_layers'], comb['size_hidden_layers']))
    # da qua in poi definisco il modello
    model = core.modules.Model(layers)
    optimizer = cm.SGD(comb['learning_rate'], comb['decay'], comb['momentum'])
    loss = cm.MSELoss(const=1., reduction='mean')
    model.compile(optimizer, loss, metrics=[MEE(), RMSE()])
    #qua dovrei chiamare la K-fold per splittare il dataset
    train_dataset = []
    eval_dataset = []
    train_dataloader = DataLoader(train_dataset, batch_size=comb['size_minibatch'], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
    #poi devo chiamare K volte model.train, giusto? intanto facciamo una
    history = model.train(
        train_dataloader, eval_dataloader, n_epochs=comb['max_epoch'],
        callbacks=[
            EarlyStopping('Val_MEE', patience=comb['patience']),
            TrainingCSVLogger()
        ]
)