import torch
import os
from torch.utils.data import Dataset
from utils.point_cloud import rotate_points_to_pos,normalize_pc
from utils.direction import get_cp1
from utils.distance import geodesic_distance_between_R
    

class AssembleDataset(Dataset):
    def __init__(self,dir,split,train_num:int=10000,val_num=3000,use_normals=True,normalize=False,horizon=1):
        self.dir=dir
        self.data = self.make_dataset(split,train_num,val_num)
        self.horizon=horizon
        self.normalize = normalize
        self.use_normal = use_normals
    
    def __len__(self):
        """
        Returns:
            int: The total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        all_finished = sample["all_finished"]
        if len(sample["interaction"]["move"])==0:
            part1_pos=sample["start_part1_pose"]
        else:
            part1_pos=sample["interaction"]["pose1"][-1]
        
        if "part2_points" in sample.keys():
            points=torch.cat((sample["part1_points"],sample["part2_points"]),dim=0)
            normals=torch.cat((sample["part1_normals"],sample["part2_normals"]),dim=0)
        else:
            points=sample["part1_points"]
            normals=sample["part1_normals"]
        normals[torch.isnan(normals).any(dim=-1)]=torch.tensor([0.0, 0.0, 0.0])
        
        if not "cp1" in sample.keys():
            cp1=get_cp1(torch.cat((sample["part2_points"],sample["part2_normals"]),dim=-1),part1_pos,task=self.dir.split("/")[-1])
            cp1_normal=cp1[3:6]
            cp1=cp1[:3]
        else:
            cp1=sample["cp1"][:3]
            cp1_normal=sample["cp1"][3:6]
            
        if torch.isnan(cp1_normal).any():
            cp1_normal=torch.tensor([0.0, 0.0, 0.0])
        cp2=sample["sampled_point"]
        cp2_normal=sample["sampled_normal"]
        if torch.isnan(cp2_normal).any():
            cp2_normal=torch.tensor([0.0, 0.0, 0.0])
        dir2=sample["action_direct"]
        
        if not sample["interaction"]["move"]:
            i_move=torch.zeros((1,12))
            i_cp2=torch.zeros((1,6))
            i_dir2=torch.zeros((1,3))
            i_pcs=torch.zeros((1,2048,6))
        else:
            i_move=torch.stack(sample["interaction"]["move"])
            i_cp2=torch.stack(sample["interaction"]["cp2"])
            i_dir2=torch.stack(sample["interaction"]["dir2"])
            i_pcs=torch.stack(sample["interaction"]["pcs"])
            
        # i_move=sample["interaction"]["move"]
        # i_cp2=sample["interaction"]["cp2"]
        # i_dir2=sample["interaction"]["dir2"]
        # i_pcs=sample["interaction"]["pcs"]
        i_cp2[:,3:6][torch.isnan(i_cp2[:,3:6]).any(dim=-1)]=torch.tensor([0.0, 0.0, 0.0])
        i_pcs[:,:,3:6][torch.isnan(i_pcs[:,:,3:6]).any(dim=-1)]=torch.tensor([0.0, 0.0, 0.0])
        if self.normalize:
            i_T=torch.tensor(sample["interaction"]["pose1"])
            for i in range(len(i_move)):
                pre_p=torch.cat((i_pcs[i][:,:3],i_cp2[i][:,:3]),dim=0)
                pre_n=torch.cat((i_pcs[i][:,3:6],i_cp2[i][3:6],i_dir2[i]),dim=0)
                T=i_T[i]
                pre_p=rotate_points_to_pos(pre_p,T)
                pre_p,_,_=normalize_pc(pre_p)
                pre_n=pre_n@T[:3,:3]
                
                i_pcs[i][:,:3]=pre_p[:-1,:]
                i_cp2[i][:3]=pre_p[-1,:]
                i_pcs[i][:,3:6]=pre_n[:-2,:]
                i_cp2[i][3:6]=pre_n[-2,:]
                i_dir2[i][:,3]=pre_n[-1,:]
            
            if len(i_move)<=0:
                T=sample["start_part1_pose"]
            else:
                T=sample["interaction"]["pose2"][-1]
            pre_p=torch.cat((points,cp1,cp2),dim=0)
            pre_n=torch.cat((normals[:,:3],cp1_normal,cp2_normal,dir2),dim=0)
            pre_p=rotate_points_to_pos(pre_p,T)
            pre_p,_,_=normalize_pc(pre_p)
            pre_n=pre_n@T[:3,:3]
            
            points=pre_p[:-2,:]
            cp1=pre_p[-2,:]
            cp2=pre_p[-1,:]
            normals=pre_n[:-3,:]
            cp1_normal=pre_n[-3,:]
            cp2_normal=pre_n[-2,:]
            dir2=pre_n[-1,:]
        
        if len(i_move) < self.horizon:
            pading_m = torch.zeros((self.horizon-len(i_move),12))
            pading_c = torch.zeros((self.horizon-len(i_move),6))
            pading_d = torch.zeros((self.horizon-len(i_move),3))
            pading_p = torch.zeros((self.horizon-len(i_move),2048,6))
            
            i_move =  torch.cat((pading_m,i_move),dim=0)[:self.horizon]
            i_cp2 =  torch.cat((pading_c,i_cp2),dim=0)[:self.horizon]
            i_dir2 =  torch.cat((pading_d,i_dir2),dim=0)[:self.horizon]
            i_pcs =  torch.cat((pading_p,i_pcs),dim=0)[:self.horizon]
           
        if all_finished:
            reward = 1-sample["distance"]  
            if "drawer" in self.dir:
                part1_pos=sample["final_part1_pose"][:2,3]
                part2_pos=sample["final_part2_pose"][:2,3]
                dis=torch.norm(part1_pos-part2_pos)
                if "pull" in self.dir:
                    reward = 1- sample["distance"] - max(0,0.2-dis)*2
                else:
                    reward = (0.2-sample["distance"])*5/2+ (0.5-dis)
            elif "bucket" in self.dir:
                if sample["distance"] <10.0:
                    height = sample["final_part1_pose"][2,3] - sample["start_part1_pose"][2,3]
                    reward = 1.0 -(0.15 -height)*10-geodesic_distance_between_R(sample["start_part1_pose"][:3,:3],sample["final_part1_pose"][:3,:3])*10
                    
            reward = torch.clamp(reward,0.0,1.0).squeeze()    
        else:
            raise ValueError("This should not happen")
        
        points=torch.cat((points,normals),dim=-1)
        cp1=torch.cat((cp1,cp1_normal),dim=-1)
        cp2=torch.cat((cp2,cp2_normal),dim=-1)    
        
        # print("idx:",idx)
        # print("points:",points.shape)
        # print("cp1:",cp1.shape)
        # print("cp2:",cp2.shape)
        # print("dir2:",dir2.shape)
        # print("i_cp2:",i_cp2.shape)
        # print("i_dir2:",i_dir2.shape)
        # print("i_pcs:",i_pcs.shape)
        # print("i_move:",i_move.shape)
        # print("reward:",reward.shape)
        
        if self.use_normal:
            return points,cp1,cp2,dir2,(i_cp2,i_dir2,i_pcs,i_move),reward
        else:
            return points[:,:3],cp1[:3],cp2[:3],dir2[:3],(i_cp2[:,:3],i_dir2,i_pcs[:,:,:3],i_move),reward
    
    
    def make_dataset(self,split,train_num,val_num):
        dataset = []
            
        if split == "train":
            # check cache
            if os.path.exists(os.path.join(self.dir, split + "_" + str(train_num)+"_cl" + ".pt")):
                print("loading from cache")
                return torch.load(os.path.join(self.dir,split + "_" + str(train_num)+"_cl" + ".pt"), weights_only=True)
            for i in range(train_num):
                
                #_data = torch.load(os.path.join(self.dir,"finished","data_" + str(i) + ".pt"), weights_only=True)
                try:
                    _data = torch.load(os.path.join(self.dir,"finished", "data_" + str(i) + ".pt"), weights_only=True)
                    _data["idx"]= torch.tensor(i)
                except:
                    print(f"Failed to load data_{i}.pt")
                #     continue
                dataset.append(_data)
        else:
            # check cache
            if os.path.exists(os.path.join(self.dir, split + "_" + str(val_num)+"_cl" + ".pt")):
                print("loading from cache")
                return torch.load(os.path.join(self.dir, split + "_" + str(val_num)+"_cl" + ".pt"), weights_only=True)
            
            for i in range(train_num, train_num + val_num):
                _data = torch.load(os.path.join(self.dir, "finished","data_" + str(i) + ".pt"), weights_only=True)
                _data["idx"]= torch.tensor(i)
                # try:
                #     _data = torch.load(os.path.join(self.dir, "data_" + str(i) + ".pt"), weights_only=True)
                # except:
                #     print(f"Failed to load data_{i}.pt")
                #     continue
                dataset.append(_data)
      
        # cache the dataset
        if split == "train":
            torch.save(dataset, os.path.join(self.dir, split + "_" + str(train_num) +"_cl"+ ".pt"))
        else:
            torch.save(dataset, os.path.join(self.dir, split + "_" + str(val_num)+"_cl"+ ".pt"))
        
        return dataset