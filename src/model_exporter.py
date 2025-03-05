from pathlib import Path
import joblib

class ModelExporter:
    def __init__(self):
        pass

    def create_folder(self, folder_location):
        '''This function creates a folder at a desired location
        Parameter:
        folder_location: file path of the folder

        Returns: a new folder
        '''
        Path(folder_location).mkdir(exist_ok=True)

    def export_model(self, trained_model, model_name:str):
        '''This function will export a trained model into the desired location
        Parameter:
        trained_model: model that you want to export
        model_name: file name of the model

        Returns:
        A saved .pkl at model folder of this pipeline
        '''
        joblib.dump(trained_model, f'../aiapbg10-Chan-Guan-Ling-162D/model/{model_name}.pkl')