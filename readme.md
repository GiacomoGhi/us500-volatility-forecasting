# US500 Volatility forecasting ()

## Start-up

Direttamente dopo la copia del repository è possibile lanciare forecasting.py.
Lo script andrà a:

1. Fare il preprocessing del dataset D1.
2. Mostrare tre tipi di analisi statistiche dei dataset presenti in ./processed-data
3. La visualizzazione della preview dell'ultimo modello caricato come 'best model'
4. Iniziare l'allenamento di un nuovo modello.

N.B.: L'allenamento della rete richide circa 5 min con un processore Intel i7 - 11th gen

## Contesto

Uno dei principali interessi di un investitore è la gestione del rischio.
Conoscere quanta volatilità aspettarsi a mercato permette di capire "quanto sono agitate le acque".

La rete neurale qui presente ha l'obbiettivo di predirre la volatilità attesa in un determinato periodo di tempo (time frame) futuro.

Sono presenti tre dataset per l'addestramento su tre time frame diversi.
E' possibile selezionare il time frame sul quale addestrare la rete dal file di configurazione (io.time_frame).

Tutti i dataset in questione sono composti da data, orario e valori di apertura, massimo, minimo e chiusura (ohlc) del prezzo.
Avremo quindi dataset distinti per i valori ohlc giornalieri, orari ed a 15 minuti

## Architettura

1. Tipologia rete: LSTM
2. Funzione di attivatione: Lineare (net_1);
3. Funzione di loss: MSE
4. Ottimizzatore: Adadelta

## Preprocessing dei dati

I dataset possiedono solo i valori o, h, l, c, data e ora.
la classe ohlcPreprocessor si occupa di:

- Calcolare ed aggiungere la volatilità del mercato, in quel time frame, come valore percentuale.
- Fare il casting dell'orario da stringa a valore intero.
- Suddividere il dataset nei sotto insiemi per addestramento (60%), validazione (20%), test (20%)

## Analisi statistiche dei Dataset

Lo script forecasting.py sfrutta la classe DatasetAnalyzer ( from dataset_analyzer.py ) per visualizzare tre tipi di analisi statistiche dei dataset utilizzati:

1. plot_last_column_distribution() vistualizza la distribuzione dei valori di volatilità
2. plot_average_volatility_by_day_of_week() visualizza l'istogramma dei valori di volatilità media raggrupata per giorni della settimana
3. plot_average_volatility_by_hour() visualizza l'istogramma dei valori di volatilità media raggrupata per le ore del giorno

Da notare, nel risultato di plot_average_volatility_by_hour, per i dataset contenenti i valori orari ed a 15 minuti, il picco di volatilità
attorno alle 14-15 (CET) del giorno. Orario che coincide con l'apertura della borsa di New York

## AutoOptimizer

La classe AutoOptimizer automatizza l'ottimizzazione dei parametri di un modello.
Carica una configurazione iniziale e genera automaticamente tutte le combinazioni possibili dei perparametri specificati per l'ottimizzazione.
Essa esegue poi training e testing per ogni combinazione e salva la configurazione con la minor perdita di test.
Infine, la classe stampa a console la configurazione migliore e crea un grafico che confronta i dati previsti con quelli reali.

Per utilizzare l'AutoOptimizer è necessario:

- mettere a true la proprietà io.use_auto_optimizer;
- selezionare i valori in "net_parameters" da ottimizzare, mettendo a true la proprietà "optimize"
- per i valori selezionari, specificare il range di valori da testare e lo step con il quale l'ottimizzatore andrà ad avanzare da inizio a fine del range.

## Esperimenti effettuati e Risultati

Sono stati eseguiti 4 ottimizzazioni sfruttando la classe AutoOptimizer:

1. Nel primo esperimento è stata fatta l'ottimizzazione per i parametri relativi all'architettura della rete.
   Sfruttando l'AutoOptimizer sono stati eseguiti 75 esperimenti con 75 diverse combinazioni riguardo il tipo di cella, il valore dei hidden_size e num_layers.
   Tra le possibili combinazioni, la loss minore è stata generata usando i seguenti valori: hidden_size.value = 16; num_layers.value = 1, cell_type.value = 2
   (il valore 2 di cell_type corrisponde all'utilizzo di celle RNN)

2. Il secondo esperimento è stato incentrato nella ricerca del miglior valore di batch_size. Provando con valori da 1 a 50, con step di 5.
   Il valore di batch_size che ha generato la loss minore è stato uguale a 31

3. Il secondo esperimento è stato incentrato nella ricerca del miglior valore di window_size. Provando con valori da 10 a 100, con step di 10.
   Il valore di window_size che ha generato la loss minore è stato uguale a 20

4. Il secondo esperimento è stato incentrato nella ricerca del miglior valore di epochs. Provando con valori da 1 a 25, con step di 5.
   Il valore di epochs che ha generato la loss minore è stato uguale a 21

## Considerazioni finali

Dalla tensorboard è possibile visualizzare l'andamento della loss. A differenza degli andamenti visti a lezione, aventi una forma di una parabola discendente,
quelli prodotti da questa rete hanno un'andamento apparentemente casuale. Osservando il grafico prodotto a fine addestramento però
