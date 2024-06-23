# US500 Volatility forecasting ()

## Start-up

Direttamente dopo la copia del repository è possibile lanciare forecasting.py.
Lo script andrà a:

1. Fare il preprocessing del dataset D1.
2. Mostrare una piccola analisi statistica generando il grafico della distribuzione dei valori di volatilità
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
2. Funzione di attivatione: Lineare (net_1); ReLU(net_2)
3. Funzione di loss: MSE
4. Ottimizzatore: Adadelta

## Preprocessing dei dati

I dataset possiedono solo i valori o, h, l, c, data e ora.
la classe ohlcPreprocessor si occupa di:

- Calcolare ed aggiungere la volatilità del mercato, in quel time frame, come valore percentuale.
- Fare il casting dell'orario da stringa a valore intero.
- Suddividere il dataset nei sotto insiemi per addestramento (60%), validazione (20%), test (20%)

## Risultati

### net_1

Risultati mediocri ma promettenti.
Attualmente i valori di volatilità mostrano un andamento molto simile ad i valori reali ma con una scala inferiore rispetto ad i valori reali.

### net_2

Risultati molto inferiori rispetto net_1, non ho quindi approfondito la sua ottimizzazione.

## Considerazioni finali

La rete net_1 mostra del potenziale per quanto riguarda il dataset di addestramento D1.

Un miglioramento notevole è stato ottenuto normalizzando il valore target (la volatilità).

I dataset H1 (time frame orario) ed M15 (time frame a 15 minuti) posseggono una mole di dati veramente molto elevata rispetto ad il dataset D1 (time frame giornaliero).

L'addestramento è stato quindi ottimizzato solo per dataset D1, decisione necessaria per la salvaguardia del mio povero pc
