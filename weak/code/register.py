import world
import dataloader
import model
from pprint import pprint

dataset = dataloader.Loader(path="../data/"+world.dataset)


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("Test Topks:", world.topks)
print('===========end===================')


MODELS = {
    'EASE': model.EASE,
    'RLAE': model.RLAE,
    'DLAE': model.DLAE,
    'LAE': model.LAE,
    'DAN_EASE': model.DAN_EASE,
    'DAN_RLAE': model.DAN_RLAE,
    'DAN_DLAE': model.DAN_DLAE,
    'DAN_LAE': model.DAN_LAE,
    'ASPIRE_RLAE': model.ASPIRE_RLAE,
    'ASPIRE_EASE': model.ASPIRE_EASE,
    'ASPIRE_DLAE': model.ASPIRE_DLAE,
    'ASPIRE_LAE': model.ASPIRE_LAE,
    # Compatibility
    'EDLAE': model.EDLAE,
    'RDLAE': model.RDLAE,
    'GFCF': model.GFCF,
}