from FastSAM.fastsam import FastSAM, FastSAMPrompt

model = FastSAM('FastSAM/weights/FastSAM-x.pt')
IMAGE_PATH = 'input/train_0012.jpg'
DEVICE = 'cuda:0'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
ann = prompt_process.everything_prompt()

prompt_process.plot(annotations=ann,output_path='./output/train_0012.jpg',)
