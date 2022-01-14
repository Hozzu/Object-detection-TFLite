from bounding_box import BBType
from coco_evaluator import get_coco_summary
from converter import coco2bb
import sys

arg_num = len(sys.argv)
if arg_num != 3 :
    print("ERROR: Argument must be 2")
    print("The first arg is ground truth annotation file and the second arg is detected annotation file")

gts = coco2bb(sys.argv[1], BBType.GROUND_TRUTH)
dts = coco2bb(sys.argv[2], BBType.DETECTED)

if len(gts) == 0 or len(dts) == 0 :
    print("ERROR: Argument must be coco json format")  

res = get_coco_summary(gts, dts)

print("mAP: {:.3f}".format(res['AP']))
print("mAP50: {:.3f}".format(res['AP50']))
print("mAP75: {:.3f}".format(res['AP75']))