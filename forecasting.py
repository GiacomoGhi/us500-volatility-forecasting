from net_runner import NetRunner
from config_helper import check_and_get_configuration


if __name__ == "__main__":

    #TODO add preprocessing
    #TODO in preprocessor, adds folder struct check and creation
    
    # Carica il file di configurazione, lo valido e ne creo un oggetto a partire dal json.
    cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')

    # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
    runner = NetRunner(cfg_obj)

    # Se richiesto, eseguo il training.
    if cfg_obj.parameters.train:
        runner.train()

    # Se richiesto, eseguo il test.
    if cfg_obj.parameters.test:
        runner.test(preview=True, print_loss=True)