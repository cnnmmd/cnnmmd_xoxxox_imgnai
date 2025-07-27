import random
from novelai_api.ImagePreset import ImageModel, ImagePreset, ImageResolution, ImageSampler, UCPreset
from xoxxox.naiapi.boilerplate import API
from xoxxox.shared import Custom

#---------------------------------------------------------------------------

class ImgPrc:
  def __init__(self, config="xoxxox/cnfimg_nai_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.nmodel = eval(diccnf["nmodel"]) # ImageModel.Anime_v3

  def status(self, config="xoxxox/cnfimg_nai_000", **dicprm):
    diccnf = Custom.update(config, dicprm)
    self.samplr = eval(diccnf["samplr"]) # ImageSampler.k_euler / ImageSampler.k_dpmpp_2m
    self.resimg = eval(diccnf["resimg"]) # ImageResolution.Small_Square / ImageResolution.Normal_Portrait_v3
    self.presng = eval(diccnf["presng"]) # UCPreset.Preset_None / UCPreset.Preset_Light / UCPreset.Preset_Heavy
    self.flgsme = eval(diccnf["flgsme"]) # True / False

  async def infere(self, prompt, promng):
    async with API() as hdlapi:
      apinai = hdlapi.api
      preset = ImagePreset.from_default_config(self.nmodel)

      sedmin = 1
      sedmax = 4294967296 # 2^32
      preset.steps = 28
      preset.scale = 5
      preset.quality_toggle = True
      preset.n_samples = 1

      preset.seed = random.randint(sedmin, sedmax) # 1 / 4294967296
      preset.sampler = self.samplr
      preset.smea = self.flgsme
      preset.resolution = self.resimg
      preset.uc_preset = self.presng
      preset.uc = promng

      async for _, imgout in apinai.high_level.generate_image(prompt, self.nmodel, preset):
        return imgout
