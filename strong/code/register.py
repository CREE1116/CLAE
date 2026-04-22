import world
import dataloader
import model
from pprint import pprint


dataset = dataloader.Loader(path="../data/" + world.dataset)


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
    'DAspire_EASE': model.DAspire_EASE,
    'DAspire_LAE': model.DAspire_LAE,
    'DAspire_RLAE': model.DAspire_RLAE,
    'DAspire_DLAE': model.DAspire_DLAE,
    # Legacy / Compatibility
    'EDLAE': model.EDLAE,
    'RDLAE': model.RDLAE,
    'GFCF': model.GFCF,
    'EASE_DAN': model.EASE_DAN,
    'IPS_LAE': model.IPS_LAE,
    'IPS_EASE': model.IPS_EASE,
    'IPS_RLAE': model.IPS_RLAE,
    'IPS_DLAE': model.IPS_DLAE,
}