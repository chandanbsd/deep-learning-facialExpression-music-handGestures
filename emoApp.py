import  argparse, os


parser = argparse.ArgumentParser(description="Options")
parser.add_argument("--update", help="Update Model", action="store_true")
args = parser.parse_args()

if args.update:
	os.system('python emoTrain.py')
    
else:
    os.system('python emoDetect.py')