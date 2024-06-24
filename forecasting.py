from auto_optimizer import AutoOptimizer
from dataset_analyzer import DatasetAnalyzer
from net_runner import NetRunner
from config_helper import check_and_get_configuration
from preprocessor import ohlcDataPreprocessor


if __name__ == "__main__":
    
    # Carica il file di configurazione, lo valido e ne creo un oggetto a partire dal json.
    cfg_obj = check_and_get_configuration('./config/config.json', './config/config_schema.json')

    # preprocessore per calcolare la volatilità dai dati di open, high, low, close 
    ohlcDataPreprocessor(cfg_obj.io.D1.non_processed_dataset, "./processed-data/D1")
    ohlcDataPreprocessor(cfg_obj.io.H1.non_processed_dataset, "./processed-data/H1")
    ohlcDataPreprocessor(cfg_obj.io.M15.non_processed_dataset, "./processed-data/M15")
    
    # Analisi statistiche della volatilità di mercato 
    analyzer = DatasetAnalyzer("./processed-data/", "train.csv")
    
    analyzer.plot_last_column_distribution()
    analyzer.plot_average_volatility_by_day_of_week()
    analyzer.plot_average_volatility_by_hour()
    
    # Se l'opzione di utilizzare l'ottimizzatore automatico è abilitata
    if cfg_obj.io.use_auto_optimizer:    
        autoOptimizer = AutoOptimizer(cfg_obj)
        
        # Esegue le ottimizzazioni automatiche
        autoOptimizer.run_optimizations()
    
    else:
        # Creo l'oggetto che mi permettera' di addestrare e testare il modello.
        runner = NetRunner(cfg_obj)

        # Se richiesto, eseguo il training.
        if cfg_obj.parameters.train:
            runner.train()

        # Se richiesto, eseguo il test.
        if cfg_obj.parameters.test:
            runner.test(preview=True, print_loss=True)
