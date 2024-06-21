import datetime
import sys
import time
import math
import shutil
import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from visual_util import ColoredPrint as cp
from custom_dataset_csv import CustomDatasetCsv

from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
np.random.seed(42)


class NetRunner():

    def __init__(self, cfg_object: object) -> None:
        
        cp.purple('Initializing net runner...')
        
        # Salvo il file di configurazione.
        self.cfg = cfg_object

        # Dal file di configurazione seleziono il files per il data set e per l'output
        if (self.cfg.io.time_frame == "D1"): 
            self.data_files_dirs = self.cfg.io.D1
        elif (self.cfg.io.time_frame == "H1"):
            self.data_files_dirs = self.cfg.io.H1
        else:
            self.data_files_dirs = self.cfg.io.M15            
        
        # Acquisisco la rete, in base al tipo richiesto.
        self.net = self.__get_net()
        
        # Carico e predispongo i loader dei dataset.
        self.__load_data()

        # Predispone la cartella di output.
        self.out_root = Path(self.data_files_dirs.out_folder)
        
        # Il percorso indicato esiste?
        if not self.out_root.exists():
            cp.cyan(f'Creating output directory: {self.out_root}')
            self.out_root.mkdir()
        
        # Indico dove salvero' il modello addestrato.
        self.last_model_outpath_sd = self.out_root / 'last_model_sd.pth'
        self.last_model_outpath = self.out_root / 'last_model.pth'
        self.best_model_outpath_sd = self.out_root / 'best_model_sd.pth'
        self.best_model_outpath = self.out_root / 'best_model.pth'
        
        # Se richiesto, si cerca l'ultimo modello salvato in fase di addestramento.
        # Di lui ci interessa lo stato dei pesi, lo state_dict.
        # Se presente, la rete sara' inizializzata a quello stato.
        if self.cfg.train_parameters.reload_last_model:
            try:
                self.net.load_state_dict(torch.load(self.last_model_outpath_sd))
                cp.green('Last model state_dict successfully reloaded.')
            except:
                cp.red('Cannot reload last model state_dict.')
        
        # Funzione di costo.
        cp.cyan(f'Created loss function.')
        self.criterion = nn.MSELoss()
                
        # Adadelta non necessita dell'uso di un learning rate iniziale
        self.optimizer = optim.Adadelta(
            self.net.parameters()
        )

    def train(self) -> None:
        
        cp.purple("Training...")

        # Inizializza un writer per loggare dati nella tensorboard.
        exp_name = 'exp_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        cp.blue("tesorboard_exp_folder: " + self.cfg.io.tesorboard_exp_folder)
        cp.blue("exp_name: " + exp_name)
        lab_path = Path(self.cfg.io.tesorboard_exp_folder)
        exp_path = lab_path / exp_name
        writer = SummaryWriter(exp_path)
        
        
        # Logga il modello nella tensorboard.
        sample_tensor = torch.randn(
            size=(
                self.cfg.hyper_parameters.batch_size,
                self.cfg.hyper_parameters.window_size,
                7
            ), 
            dtype=torch.float32
        )
        writer.add_graph(self.net, sample_tensor)

        # Conteggio degli step totali.
        global_step = 0
        
        epochs = self.cfg.hyper_parameters.epochs
        cp.cyan(f'Training loop epochs: {epochs}')

        # Salvo in una variabile in modo da mostrare una sola volta.
        show_preview = self.cfg.parameters.show_preview
        
        # Ogni quanto monitorare la funzione di costo.
        train_step_monitor = self.cfg.train_parameters.step_monitor
        loss_target = self.cfg.train_parameters.loss_target
        
        cp.cyan(f'Training monitor every {train_step_monitor} steps, preview: {show_preview}.')
        cp.cyan(f'Training will stop when reaching loss target ({loss_target}) for train and validation data.')
        
        es_start_epoch = self.cfg.early_stop_parameters.start_epoch
        es_loss_evaluation_epochs = self.cfg.early_stop_parameters.loss_evaluation_epochs
        es_patience = self.cfg.early_stop_parameters.patience
        es_improvement_rate = self.cfg.early_stop_parameters.improvement_rate
        
        cp.cyan(f'Early stop check will start at epoch {es_start_epoch}.')
        cp.cyan(f'Validation loss evaluated every {es_loss_evaluation_epochs} epochs.')
        cp.cyan(f'Early stop will be triggered after {es_patience} epochs of no improvement.')
        cp.cyan(f'Minimum requested improvement is {es_improvement_rate}% on validation loss.')

        tr_losses_x, tr_losses_y = [], []           # Raccoglitori loss di training.
        tr_run_losses_x, tr_run_losses_y = [], []   # Raccoglitori loss di training ogni step del monitor.
        va_losses_x, va_losses_y = [], []           # Raccoglitori loss di validazione.

        best_tr_loss = float('inf') # Traccia la migliore loss raggiunta in training.
        best_va_loss = float('inf') # Traccia la migliore loss raggiunta in validazione.
        
        # Con questo contatore, si valuta per quanti check consecutivi
        # la loss di validazione non e' migliorata.
        va_loss_no_improve_ep_ctr = 0

        reached_loss_target = False     # FLAG EVENTO: sono state raggiunte le loss target?
        early_stop_check = False        # FLAG EVENTO: puo' iniziare il check regolare per l'early stop.
        early_stop = False              # FLAG EVENTO: l'early stop e' scattato, stop dell'addestramento.

        # Loop di addestramento per n epoche.
        for epoch in range(epochs):
            
            if reached_loss_target:
                cp.green('Stopping: reached loss target!')
                break
            
            # L'analisi dell'early stop inizia solo quando sono passate le epoche richieste.
            if (epoch + 1) == es_start_epoch:
                early_stop_check = True
            
            # Se l'early stop e' scattato, ci si ferma.
            if early_stop:
                cp.yellow('Stopping: detected EarlyStop!')
                break
            
            current_tr_loss = 0
            current_tr_loss_counter = 0

            monitor_loss = 0.0

            # Stop di addestramento. Dimensione batch_size.
            for i, data in enumerate(self.tr_loader, 0):

                # Le rete entra in modalita' addestramento.
                self.net.train()

                # Per ogni input tiene conto della sua etichetta.
                sequence, next_element = data
                
                if show_preview:
                    self.test(self.tr_b1_loader, use_current_net=True, preview=True)
                    cp.blue('...for training preview.')
                    show_preview = False

                # L'input attraversa al rete. Errori vengono commessi.
                outputs = self.net(sequence)

                # Calcolo della funzione di costo sulla base di predizioni e previsioni.
                next_element = next_element.unsqueeze(1)
                loss = self.criterion(outputs, next_element)
                
                # I gradienti vengono azzerati.
                self.optimizer.zero_grad()

                # Avviene il passaggio inverso.
                loss.backward()
                
                # Passo di ottimizzazione
                self.optimizer.step()

                # Monitoraggio statistiche.
                monitor_loss += loss.item()
                current_tr_loss += loss.item()
                current_tr_loss_counter += 1
                
                if (i + 1) % train_step_monitor == 0:
                    tr_run_losses_y.append(monitor_loss / train_step_monitor)
                    tr_run_losses_x.append(global_step)
                    
                    print(f'global_step: {global_step:5d} - [ep: {epoch + 1:3d}, step: {i + 1:5d}] loss: {loss.item():.6f} - running_loss: {(monitor_loss / train_step_monitor):.6f}')
                    
                    monitor_loss = 0.0

                    # Logga parametri nella tensorboard
                    writer.add_scalar('train/loss', loss.item(), i)
                
                tr_losses_y.append(loss.item())
                tr_losses_x.append(global_step)

                global_step += 1
                
            tr_loss = current_tr_loss / current_tr_loss_counter
            
            if tr_loss < best_tr_loss:
                # Calcolo il tasso di miglioramento.
                improve_ratio = abs((tr_loss / best_tr_loss) - 1) * 100
                cp.green(f'... training loss improved: {best_tr_loss:.6f} --> {tr_loss:.6f} ({improve_ratio:.3f}%)')
                best_tr_loss = tr_loss
                
                torch.save(self.net.state_dict(), self.best_model_outpath_sd)
                torch.save(self.net, self.best_model_outpath)
                cp.yellow('Best model saved.')
                
            # Controllo della loss di validazione:
            # - Se il check dell'early stop e' abilitato.
            # - E sono passate le epoche di attesa fra un check e l'altro.     
            if early_stop_check and (epoch + 1) % es_loss_evaluation_epochs == 0:
                
                cp.cyan("... Evaluating validation loss ...")
                va_loss = self.test(self.va_b1_loader, use_current_net=True)
                va_losses_x.append(global_step)
                va_losses_y.append(va_loss)
                
                # Verifica se lo loss di validazione e' migliorata:
                # - Se non lo e', aggiurno il counter dei NON MIGLIORAMENTI.
                # - Se lo e' ma non a sufficienza, aggiurno il counter dei NON MIGLIORAMENTI.
                # - Se lo e', azzero il counter.
                if va_loss < best_va_loss:
                    
                    # Calcolo il tasso di miglioramento.
                    improve_ratio = abs((va_loss / best_va_loss) - 1) * 100
                    
                    # Verifico che il miglioramento non sia inferiore al tasso.
                    if improve_ratio >= es_improvement_rate:
                        cp.green(f'... validation loss improved: {best_va_loss:.6f} --> {va_loss:.6f} ({improve_ratio:.3f}%)')
                        best_va_loss = va_loss
                        va_loss_no_improve_ep_ctr = 0
                    else:
                        cp.red(f'... validation loss NOT improved ... ({improve_ratio:.3f}%) < ({es_improvement_rate}%)')
                        va_loss_no_improve_ep_ctr += 1
                else:
                    cp.red(f'... validation loss NOT improved')
                    va_loss_no_improve_ep_ctr += 1

            # Se la loss di validazione non migliora da 'patience' epoche, e' il
            # momento di alzare il FLAG e richiedere l'early stop.
            if va_loss_no_improve_ep_ctr >= es_patience:
                early_stop = True
            
            # Se le loss di training e validazione raggiungono il target, richiedo lo stop.    
            if best_tr_loss < loss_target and best_va_loss < loss_target:
                reached_loss_target = True

        # Salvo l'ultimo stato/modello a termine dell'addestramento.
        torch.save(self.net.state_dict(), self.last_model_outpath_sd)
        torch.save(self.net, self.last_model_outpath)
        cp.yellow('Last model saved.')
        
        cp.blue('Finished Training.')

        self.test(self.tr_b1_loader, use_current_net=True, preview=True)
        cp.blue('...of training data.')
        
        self.test(self.va_b1_loader, use_current_net=True, preview=True)
        cp.blue('...of validation data.')

        # Logga i risultati di addestramento
        writer.add_hparams(
            {
                'num_epochs' : self.cfg.hyper_parameters.epochs, 
                'batch_size': self.cfg.hyper_parameters.batch_size, 
                'window_size': self.cfg.hyper_parameters.window_size, 
                'momentum': self.cfg.hyper_parameters.momentum,
            }, 
            {
                'hparams/best_loss' : loss.item()
            }
        )
        
        writer.flush()
        writer.close()
    
    def test(self, loader: torch.utils.data.DataLoader = None, use_current_net: bool = False, preview : bool = False, print_loss: bool = False):

        cp.purple("Testing...")

        # Se non specifico diversamente, testo sui dati di test.
        if loader is None:
            loader = self.te_b1_loader

        # Se richiesto, testo sul modello corrente e non il migliore.
        if use_current_net:
            net = self.net
        else:
            # Per usare il modello migliore:
            # - Richiedo un modello 'nuovo'.
            # - Inizializzo i suoi pesi allo stato del modello migliore.
            net = self.__get_net()
            
            try:
                net.load_state_dict(torch.load(self.best_model_outpath_sd))
            except Exception as e:
                cp.red(f'Error loading model state_dict: {repr(e)}')
                

        real, pred = [], []
        current_loss = 0
        current_loss_counter = 0

        # Stop di validazione.
        for _, data in enumerate(loader, 0):

            # Le rete entra in modalita' addestramento.
            net.eval()
            
            # Per ogni input tiene conto della sua etichetta.
            sequence, next_element = data

            # Disabilita computazione dei gradienti.
            with torch.no_grad():
                
                # Esegue le predizioni.
                outputs = net(sequence)

                # Calcola la loss.
                next_element = next_element.unsqueeze(1)
                loss = self.criterion(outputs, next_element)

                current_loss += loss.item()
                current_loss_counter += 1
                
                real.append(next_element[0, -1].item())
                pred.append(outputs.item())
        
        if preview:
            x = np.linspace(0, len(real)-1, len(real))            
            _, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.plot(x, real)
            ax1.set_title('Real sequence')
            ax2.plot(x, pred)
            ax2.set_title('Predicted sequence')
            ax3.plot(x, real, label='Real')
            ax3.plot(x, pred, label='Predicted')
            ax3.set_title('Combined sequences')
            ax3.legend()
            plt.tight_layout()
            plt.show()

        loss = current_loss / current_loss_counter
        
        if print_loss:
            cp.yellow(f'Test loss: {loss:.6f}')
        
        return loss
        
    # Ottiene un oggetto 'rete' del tipo richiesto.
    def __get_net(self):
        
        if self.cfg.train_parameters.network_type.lower() == 'net_1':
            from nets.net_1 import Net
        elif self.cfg.train_parameters.network_type.lower() == 'net_2':
            from nets.net_2 import Net
        else:
            print(f'Unknown net.')
            sys.exit(-1)
            
        return Net(hidden_size=self.cfg.lstm_parameters.hidden_size,
                   num_layers=self.cfg.lstm_parameters.num_layers)
    
    # Carica i Dataset tramite Dataloader e scopre le classi del dataset.
    def __load_data(self) -> None:
    
        cp.cyan(f'Analyzing training dataset: {self.data_files_dirs.training_file}')
        tr_dataset = CustomDatasetCsv(self.data_files_dirs.training_file, 
                                      self.cfg.hyper_parameters.window_size, 
                                      debug=self.data_files_dirs.cutom_dataset_debug)
        
        cp.cyan(f'Analyzing validation dataset: {self.data_files_dirs.validation_file}')
        va_dataset = CustomDatasetCsv(self.data_files_dirs.validation_file, 
                                      self.cfg.hyper_parameters.window_size, 
                                      debug=self.data_files_dirs.cutom_dataset_debug)
        
        cp.cyan(f'Analyzing test dataset: {self.data_files_dirs.test_file}')
        te_dataset = CustomDatasetCsv(self.data_files_dirs.test_file, 
                                      self.cfg.hyper_parameters.window_size, 
                                      debug=self.data_files_dirs.cutom_dataset_debug)

        # Creo poi il dataloader che prende i dati di addestramento a batch:
        self.tr_loader = torch.utils.data.DataLoader(tr_dataset, 
                                                     batch_size=self.cfg.hyper_parameters.batch_size, 
                                                     shuffle=False)
        
        # E i dataloader che prendono i dati, un solo elemnto per volta: usati per il test/visualizzazione.
        self.tr_b1_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=1, shuffle=False)
        self.va_b1_loader = torch.utils.data.DataLoader(va_dataset, batch_size=1, shuffle=False)
        self.te_b1_loader = torch.utils.data.DataLoader(te_dataset, batch_size=1, shuffle=False) 