from numpy import copy

class AutoOptimizer:
  def __init__(self, cfg_object) -> None:
    self.cfb = cfg_object
    self.configurations = []
    self.configContainer = []
    self.itemsCount = 0

    if (cfg_object.auto_optimizer_parameters.batch_size.enabled):
      self._addValueToOptimize(1)

    if (cfg_object.auto_optimizer_parameters.window_size.enabled):
      self._addValueToOptimize(2) 

    if (cfg_object.auto_optimizer_parameters.epochs.enabled):
      self._addValueToOptimize(3)

  
  def _addValueToOptimize(self, itemId: int):
    if (itemId == 1):
      self.configContainer.inser(copy.copy(self.cfb.auto_optimizer_parameters.batch_size))
    
    elif (itemId == 2):
      self.configContainer.inser(copy.copy(self.cfb.auto_optimizer_parameters.window_size))
    
    elif (itemId == 3):
      self.configContainer.inser(copy.copy(self.cfb.auto_optimizer_parameters.epochs))

  
  def _getPossibleConfigurations(self):


    if len(self.configContainer) == 1:
      item = self.configContainer[0]
      for i in range(item.start, item.end, item.step):
        #item.hyper_parameters.batch_size = i
        #add value to config and schema, also then adjust net
        self.configurations.insert(newConfig)
      
    elif len(self.configContainer) == 2:
      
    elif len(self.configContainer) == 3:


def getListOfPossibleConfiguration(self, cfg_object: object) -> None:
  hParams = cfg_object.auto_optimizer_parameters

  configurations = []

  if (hParams.batch_size.enabled and hParams.window_size.enabled and hParams.epochs.enabled):
      b = hParams.batch_size
      
      w = hParams.window_size
      
      e = hParams.epochs
      
      for i in range(b.start, b.end, b.step):
        for j in range(w.start, w.end, w.step):
          for k in range(e.start, e.end, e.step):
            newConfig = copy.copy(cfg_object)

            newConfig.hyper_parameters.batch_size = i
            newConfig.hyper_parameters.window_size = j
            newConfig.hyper_parameters.epochs = k

            configurations.insert(newConfig)
  
  
  elif (hParams.batch_size.enabled and hParams.window_size.enabled):
      b = hParams.batch_size
      
      w = hParams.window_size
      
      for i in range(b.start, b.end, b.step):
        for j in range(w.start, w.end, w.step):
          newConfig = copy.copy(cfg_object)

          newConfig.hyper_parameters.batch_size = i
          newConfig.hyper_parameters.window_size = j

          configurations.insert(newConfig)
  
  
  elif (hParams.batch_size.enabled and hParams.window_size.enabled):
      b = hParams.batch_size
      
      w = hParams.window_size
      
      for i in range(b.start, b.end, b.step):
        for j in range(w.start, w.end, w.step):
          newConfig = copy.copy(cfg_object)

          newConfig.hyper_parameters.batch_size = i
          newConfig.hyper_parameters.window_size = j

          configurations.insert(newConfig)
  
  
  elif (hParams.batch_size.enabled):
      b = hParams.batch_size
      
      for i in range(b.start, b.end, b.step):
        newConfig = copy.copy(cfg_object)

        newConfig.hyper_parameters.batch_size = i

        configurations.insert(newConfig)

  else: 
    newConfig = copy.copy(cfg_object)
    configurations.insert(newConfig)