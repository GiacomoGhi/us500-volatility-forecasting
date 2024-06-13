from net_runner import NetRunner
from config_helper import check_and_get_configuration
from plot_last_column_distribution import plot_last_column_distribution
from preprocessor import ohlcDataPreprocessor


if __name__ == "__main__":

    #TODO add preprocessing
    #TODO in preprocessor, adds folder struct check and creation
    
    # Carica il file di configurazione, lo valido e ne creo un oggetto a partire dal json.
    cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')

    
    # Dal file di configurazione seleziono il files per il data set e per l'output
    data_path = ""
    preprocessor_output_dir = "./processed-data/" + cfg_obj.io.time_frame

    if (cfg_obj.io.time_frame == "D1"): 
        data_path = cfg_obj.io.D1.non_processed_dataset
    elif (cfg_obj.io.time_frame == "H1"):
        data_path = cfg_obj.io.H1.non_processed_dataset
    else:
        data_path = cfg_obj.io.M15.non_processed_dataset

    # preprocessore per calcolare la volatilità dai dati di open, high, low, close 
    ohlcDataPreprocessor(data_path, preprocessor_output_dir)

    plot_last_column_distribution(preprocessor_output_dir + "/train.csv")

    # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
    runner = NetRunner(cfg_obj)

    # Se richiesto, eseguo il training.
    if cfg_obj.parameters.train:
        runner.train()

    # Se richiesto, eseguo il test.
    if cfg_obj.parameters.test:
        runner.test(preview=True, print_loss=True)