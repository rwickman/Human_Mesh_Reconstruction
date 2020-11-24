import cv2, pickle, trimesh
import numpy as np
import trimesh.transformations as trans

class Render:
    def __init__(self, args,  focal_len=5.):
        self._args = args
        self.h, self.w = self._args.img_size, self._args.img_size
        self._focal_len = focal_len
        self._load_smpl_faces()
        print("self._smpl_faces.shape", self._smpl_faces.shape)

    def _load_smpl_faces(self):
        with open(self._args.smpl_model, "rb") as f:
            smpl_model = pickle.load(f)
        self._smpl_faces = smpl_model["f"].astype(np.int32)


    def __call__(self, verts, img=None, img_size=None, bg_color=None):
        """Render smpl mesh
        Args:
            verts: [6890 x 3], smpl vertices
            img: [h, w, channel] (optional)
            img_size: [h, w] specify frame size of rendered mesh (optional)
        """

        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h, w = img_size[0], img_size[1]
        else:
            h, w = self.h, self.w

        mesh = self._create_mesh(verts)
        scene = mesh.scene()

        bg_color = np.zeros(4)

        image_bytes = scene.save_image(resolution=(w, h), background=bg_color, visible=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            
            temp_image = image.copy()[:,:,:3]
            white_pixels = ~temp_image.any(axis=2)

            for c in range(0, 3):
                img[:, :, c] = img[:, :, c] * white_pixels + image[:, :, c] * (1-white_pixels)
                
            image =img
    
        return image

    def _create_mesh(self, vertices):
        print("vertices.shape", vertices.shape)
        mesh = trimesh.Trimesh(vertices=vertices, faces=self._smpl_faces,
                               vertex_colors=[200, 255, 255, 255],
                               face_colors=[0, 0, 0, 0],
                               use_embree=False,
                               process=False)
        transform = trans.rotation_matrix(np.deg2rad(-180), [1, 0, 0], mesh.centroid)
        mesh.apply_transform(transform)

        return mesh

    def rotated(self, verts, deg, axis='y', img=None, img_size=None):
        rad = np.deg2rad(deg)

        if axis == 'x':
            mat = [rad, 0, 0]
        elif axis == 'y':
            mat = [0, rad, 0]
        else:
            mat = [0, 0, rad]

        around = cv2.Rodrigues(np.array(mat))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(new_v, img=img, img_size=img_size)