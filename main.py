from macro_utils import params_extraction,data_processing,model_params_processing,train_params_processing
from train_utils import train_linear,train_gcn
import argparse


def train(model):
    params = params_extraction(sweep=False)
    data_path = "data/celva_transl.csv"
    print("Printing Parameters")
    print("Printing Graph Parameters")
    print(f"Columns Considered : {params['graph_columns']}")
    print(f"Threshold : {params['threshold']}")
    print("Printing model parameters")
    print(f"Chosen Model : {model}")
    print(f"Hidden Size : {params['hidden_size']}")
    print("Printing Training Parameters")
    print(f"Learning Rate : {params['lr']}")
    print(f"Number of Epochs : {params['n_epochs']}")
    data_params = data_processing(data_path,params)
    model_params = model_params_processing(params)
    train_params = train_params_processing(params)
    if model == "linear":
        train_linear(model_params,train_params,data_params)
    elif model == 'graph':
        train_gcn(model_params,train_params,data_params)
        

def configuration(model):
    params = params_extraction(sweep=False)
    params = params_extraction(sweep=False)
    print("Printing Parameters")
    print("Printing Graph Parameters")
    print(f"Columns Considered : {params['graph_columns']}")
    print(f"Threshold : {params['threshold']}")
    print("Printing model parameters")
    print(f"Chosen Model : {model}")
    print(f"Hidden Size : {params['hidden_size']}")
    print("Printing Training Parameters")
    print(f"Learning Rate : {params['lr']}")
    print(f"Number of Epochs : {params['n_epochs']}")
    
def sweep(model):
    params = params_extraction(sweep=True)
    print("Printing Parameters")
    print("Printing Graph Parameters")
    print(f"Columns Considered : {params['graph_columns']}")
    print(f"Threshold : {params['threshold']}")
    print("Printing Sweep Parameters")
    print(f"Chosen Model : {model}")
    print(f"Hidden Size List : {params['hidden_size']}")
    print(f"Learning Rate List : {params['lr']}")
    print(f"Number of Epochs List : {params['n_epochs']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose mode")

    parser.add_argument('--train', action='store_true',
                        help="Train and compare models")
    parser.add_argument('--sweep', action='store_true',
                        help="Find the best Hyperparameters configuration")
    parser.add_argument('--configuration', action='store_true',
                        help="Print Model configuration")

    parser.add_argument('--model', type=str, choices=["linear", "graph"],
                        required=True,
                        help="Select model type: Linear or Graph")

    args = parser.parse_args()
    
    if args.train:
        train(model = args.model)
    elif args.configuration:
        configuration( model = args.model)
    elif args.sweep:
        sweep(model = args.model)
        


