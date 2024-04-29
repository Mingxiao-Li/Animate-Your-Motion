import os 

class DatasetCatalog:
    def __init__(self, ROOT):

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        self.VGGrounding = {   
            "target": "data.gligen_dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT, 'GROUNDING/gqa/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        self.FlickrGrounding = {
            "target": "data.gligen_dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT, 'GROUNDING/flickr/train-00.tsv'),
            ),
        }

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # 

        self.SBUGrounding = {   
            "target": "data.gligen_dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT, 'GROUNDING/sbu/train-00.tsv'),
            ),
         }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        self.CC3MGrounding = {   
            "target": "data.gligen_dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT, 'GROUNDING/CC3M/train-00.tsv'),
            ),
        }

        # self.CC3MGroundingHed = {
        #     "target": "data.gligen_dataset.dataset_hed.HedDataset",
        #     "train_params": dict(
        #         tsv_path=os.path.join(ROOT, 'GROUNDING/CC3M/train-00.tsv'),
        #         hed_tsv_path=os.path.join(ROOT, 'GROUNDING/CC3M/tsv_hed/train-00.tsv'),
        #     ),
        # }
        #
        #
        # self.CC3MGroundingCanny = {
        #     "target": "data.gligen_dataset.dataset_canny.CannyDataset",
        #     "train_params":dict(
        #         tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/train-00.tsv'),
        #         canny_tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv_canny/train-00.tsv'),
        #     ),
        # }
        #
        # self.CC3MGroundingDepth = {
        #     "target": "data.gligen_dataset.dataset_depth.DepthDataset",
        #     "train_params":dict(
        #         tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/train-00.tsv'),
        #         depth_tsv_path=os.path.join(ROOT,'GROUNDING/CC3M/tsv_depth/train-00.tsv'),
        #     ),
        # }
        #
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #
        #
        self.CC12MGrounding = {
            "target": "data.gligen_dataset.tsv_dataset.TSVDataset",
            "train_params": dict(
                tsv_path=os.path.join(ROOT, 'GROUNDING/CC12M/train-00.tsv'),
            ),
        }


        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

        self.Obj365Detection = {
            "target": "dataset.tsv_dataset.TSVDataset",
            "train_params":dict(
                tsv_path=os.path.join(ROOT,'OBJECTS365/train-00.tsv'),
            ),
        }


        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #
        # self.COCO2017Keypoint = {
        #     "target": "data.gligen_dataset.dataset_kp.KeypointDataset",
        #     "train_params":dict(
        #         image_root = os.path.join(ROOT,'COCO/images'),
        #         keypoints_json_path = os.path.join(ROOT,'COCO/annotations2017/person_keypoints_train2017.json'),
        #         caption_json_path = os.path.join(ROOT,'COCO/annotations2017/captions_train2017.json'),
        #     ),
        # }
        #
        #
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #
        # self.DIODENormal = {
        #     "target": "data.gligen_dataset.dataset_normal.NormalDataset",
        #     "train_params":dict(
        #         image_rootdir = os.path.join(ROOT,'normal/image_train'),
        #         normal_rootdir = os.path.join(ROOT,'normal/normal_train'),
        #         caption_path = os.path.join(ROOT,'normal/diode_cation.json'),
        #     ),
        # }
        #
        #
        # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #
        # self.ADESemantic = {
        #     "target": "data.gligen_dataset.dataset_sem.SemanticDataset",
        #     "train_params":dict(
        #         image_rootdir = os.path.join(ROOT,'ADE/ADEChallengeData2016/images/training'),
        #         sem_rootdir = os.path.join(ROOT,'ADE/ADEChallengeData2016/annotations/training'),
        #         caption_path = os.path.join(ROOT,'ADE/ade_train_images_cation.json'),
        #     ),
        # }





