from models import Unisolver_HeterNS
def get_model(args):
    model_dict = {
        'Unisolver_HeterNS': Unisolver_HeterNS,
    }
    return model_dict[args.model].Model(args=args).cuda()
