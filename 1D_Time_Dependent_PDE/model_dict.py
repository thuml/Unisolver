from models import Unisolver_1D

def get_model(args):
    model_dict = {
        'Unisolver_1D': Unisolver_1D
    }
    return model_dict[args.model].Model(args=args).cuda()
