from itertools import product
import core.modules as cm
from core.utils import RandomUniformInitializer, RandomNormalDefaultInitializer  # todo prova a usare anche normal
from core.metrics import MEE, RMSE
from core.callbacks import EarlyStopping, TrainingCSVLogger
from core.data import ArrayDataset, DataLoader
# todo nel fare i commenti, separa '#' e il testo con uno spazio ' '
# todo tieni a mente che in teoria EarlyStopping e regolarizzazione **NON** dovrebbero essere usate insieme
#  (non son sicuro, controlla sulla teoria)
# todo extend() con una lista si usa se vuoi aggiungere gli elementi di un'altra lista


def cross_product(inp: dict):
    # todo controlla che nel caso di tuple come valori in inp.values(), queste non siano
    #  spacchettate a loro volta (i.e.: 'size_hidden_layers': [(8, 4), (3, 2)] **NON** deve
    #  iterare come 8, 4, 3, 2 ma come (8, 4), (3, 2)
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


# qua andrebbero inseriti i valori su cui fare la search [YES]
param_of_search = {
    # todo non mettere uno spazio fra una chiave di un dizionario e il suo valore
    # todo nr_hidden_layers sarà implicito nella lunghezza di 'size_hidden_layers'
    # gli hidden layer dovrebbero poter essere di dimensioni diverse. Dobbiamo provare? [Assolutamente sì]
    'size_hidden_layers': [(4, 2), (1, 1)],  # todo ad esempio qui vanno tuple per ogni hidden layer
    'input_dim': [1],  # todo usa un solo valore qui per avere l'input dim (controlla input e output del CUP dataset)
    'output_dim': [1],  # todo usa un solo valore qui per avere l'output dim (controlla come sopra)
    'activation_function': [1],
    'learning_rate': [1],
    'decay': [1],  # todo che decay stai usando?
    'momentum': [1],
    'l2_lambda_reg': [1],  # todo ho rinominato in l2_lambda_reg per poter usare (se vogliamo) anche l1/l1l2
    'inf_rand': [1],
    'sup_rand': [1],
    # todo usa questi (converti in core.utils.initializers.RandomUniformInitializer con i due parametri di sopra)
    # todo prova a vedere se riesci a inserire sia uniform sia normal come inizializzazione (evitando di avere
    #  ripetizioni)
    'patience': [1],
    'max_epoch': [1],
    'size_minibatch': [1]
}

hyperpar_comb = cross_product(param_of_search)

for comb in hyperpar_comb:
    layers = [cm.Input()]
    # todo input dimension unita con hidden layers sizes per semplificare il for dopo
    nr_hidden_layers = len(comb['size_hidden_layers'])
    non_output_sizes = (comb['input_dim'],) + comb['size_hidden_layers']
    # questo for va aggiustato. servono le dimensioni di input e output e gli hidden layers possono avere dimensioni diverse
    for i in range(0, nr_hidden_layers):
        layers.append(cm.Dense(
            non_output_sizes[i],
            non_output_sizes[i+1],
            comb['activation_function'],
            weights_initializer=RandomUniformInitializer(low=comb['inf_rand'], high=comb['sup_rand']),
            weights_regularizer=cm.L2Regularizer(l2_lambda=comb['l2_lambda_reg']),
            biases_regularizer=cm.L2Regularizer(l2_lambda=comb['l2_lambda_reg']),
            # todo (se riesci) modifica sopra in modo da poter usare regolarizzazioni diverse
        ))
    layers.append(cm.Linear(
        non_output_sizes[-1], comb['output_dim'],
        weights_initializer=RandomUniformInitializer(low=comb['inf_rand'], high=comb['sup_rand']),
        weights_regularizer=cm.L2Regularizer(l2_lambda=comb['l2_lambda_reg']),
        biases_regularizer=cm.L2Regularizer(l2_lambda=comb['l2_lambda_reg']),
    ))
    # da qua in poi definisco il modello
    model = cm.Model(layers)
    # todo il decay del learning rate lo devi fare passando un oggetto core.modules.schedulers.Scheduler con opportuni
    #  parametri per __init__
    optimizer = cm.SGD(comb['learning_rate'], momentum=comb['momentum'])
    loss = cm.MSELoss(const=1., reduction='mean')
    model.compile(optimizer, loss, metrics=[MEE(), RMSE()])
    # qua dovrei chiamare la K-fold per splittare il dataset
    train_dataset = []
    eval_dataset = []
    # todo con delle liste non funziona, devi mettere ArrayDataset che contengono i dataset di training e validation,
    #  rispettivamente
    train_dataloader = DataLoader(train_dataset, batch_size=comb['size_minibatch'], shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset))
    # poi devo chiamare K volte model.train, giusto? intanto facciamo una
    history = model.train(
        train_dataloader, eval_dataloader, max_epochs=comb['max_epoch'],
        callbacks=[
            EarlyStopping('Val_MEE', patience=comb['patience']),
            TrainingCSVLogger()
        ]
    )  # todo per convenzione, lascia sempre una riga vuota alla fine di ogni file
