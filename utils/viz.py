import matplotlib.pyplot as plt
import numpy as np
import math
import os
import cv2


class Visualizer:
    def __init__(self, type):
        """
        Args:
            type: dataformat type, if aa, data will need to be converted to rotmat.
        """
        self.type = type
        if (self.type != "aa") and (self.type != "rotmat"):
            raise ValueError("Viz engine only supportas aa or rotmat types")

        self.SMPL_JOINTS = [
            "pelvis",
            "l_hip",
            "r_hip",
            "spine1",
            "l_knee",
            "r_knee",
            "spine2",
            "l_ankle",
            "r_ankle",
            "spine3",
            "l_foot",
            "r_foot",
            "neck",
            "l_collar",
            "r_collar",
            "head",
            "l_shoulder",
            "r_shoulder",
            "l_elbow",
            "r_elbow",
            "l_wrist",
            "r_wrist",
            "l_hand",
            "r_hand",
        ]
        self.SMPL_JOINT_MAPPING = {i: x for i, x in enumerate(self.SMPL_JOINTS)}
        self.SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        self.SMPL_PARENTS = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
        ]
        self.SMPL_NR_JOINTS = 24

        # These are the offsets stored under `J` in the SMPL model pickle file
        offsets = np.array(
            [
                [-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],
                [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],
                [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],
                [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],
                [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],
                [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],
                [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],
                [8.95999143e-02, -1.04856032e00, -3.04155922e-02],
                [-9.20120818e-02, -1.05466743e00, -2.80514913e-02],
                [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],
                [1.12937580e-01, -1.10320516e00, 8.39545265e-02],
                [-1.14055299e-01, -1.10107698e00, 8.98482216e-02],
                [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],
                [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],
                [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],
                [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],
                [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],
                [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],
                [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],
                [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],
                [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],
                [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],
                [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],
                [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02],
            ]
        )

        # Convert to compatible offsets
        self.smpl_offsets = np.zeros([24, 3])
        self.smpl_offsets[0] = offsets[0]
        for idx, pid in enumerate(self.SMPL_PARENTS[1:]):
            self.smpl_offsets[idx + 1] = offsets[idx + 1] - offsets[pid]

    def aa2rotmat(self, angle_axes):
        """
        Convert angle-axis to rotation matrices using opencv's Rodrigues formula.
        Args:
            angle_axes: A np array of shape (..., 3)

        Returns:
            A np array of shape (..., 3, 3)
        """
        orig_shape = angle_axes.shape[:-1]
        aas = np.reshape(angle_axes, [-1, 3])
        rots = np.zeros([aas.shape[0], 3, 3])
        for i in range(aas.shape[0]):
            rots[i] = cv2.Rodrigues(aas[i])[0]
        return np.reshape(rots, orig_shape + (3, 3))

    def Angles2Position(self, joint_angles):
        """
        Perform forward kinematics. This requires joint angles to be in rotation matrix format.
        Args:
            joint_angles: np array of shape (N, n_joints*3*3)

        Returns:
            The 3D joint positions as a an array of shape (N, n_joints, 3)
        """
        left_mult = False
        no_root = True
        offsets = self.smpl_offsets

        if self.type == "aa":
            joint_angles = self.aa2rotmat(joint_angles)

        assert joint_angles.shape[-1] == self.SMPL_NR_JOINTS * 9
        angles = np.reshape(joint_angles, [-1, self.SMPL_NR_JOINTS, 3, 3])
        n_frames = angles.shape[0]
        positions = np.zeros([n_frames, self.SMPL_NR_JOINTS, 3])
        rotations = np.zeros(
            [n_frames, self.SMPL_NR_JOINTS, 3, 3]
        )  # intermediate storage of global rotation matrices
        if left_mult:
            offsets = offsets[np.newaxis, np.newaxis, ...]  # (1, 1, n_joints, 3)
        else:
            offsets = offsets[np.newaxis, ..., np.newaxis]  # (1, n_joints, 3, 1)

        if no_root:
            angles[:, 0] = np.eye(3)

        for j in range(self.SMPL_NR_JOINTS):
            if self.SMPL_PARENTS[j] == -1:
                # this is the root, we don't consider any root translation
                positions[:, j] = 0.0
                rotations[:, j] = angles[:, j]
            else:
                # this is a regular joint
                if left_mult:
                    positions[:, j] = (
                        np.squeeze(
                            np.matmul(
                                offsets[:, :, j], rotations[:, self.SMPL_PARENTS[j]]
                            )
                        )
                        + positions[:, self.SMPL_PARENTS[j]]
                    )
                    rotations[:, j] = np.matmul(
                        angles[:, j], rotations[:, self.SMPL_PARENTS[j]]
                    )
                else:
                    positions[:, j] = (
                        np.squeeze(
                            np.matmul(rotations[:, self.SMPL_PARENTS[j]], offsets[:, j])
                        )
                        + positions[:, self.SMPL_PARENTS[j]]
                    )
                    rotations[:, j] = np.matmul(
                        rotations[:, self.SMPL_PARENTS[j]], angles[:, j]
                    )

        return positions[..., [0, 2, 1]]

    def compare_skeletal_frames(
        self,
        tensor1,
        title1,
        tensor2,
        title2,
        nrows=2,
        ncols=30,
        rotation=(0, -60),
        out_dir=None,
        fname=None,
    ):
        """
        Display two tensors in two rows of skeletal frames from a tensor in a grid of subplots,
        connecting joints based on the parent body model. The axes are completely hidden for a clean look,
        and the tensor names are displayed at the top of each row.
        Args:
            tensor1: First tensor of shape [num_frames, num_joints, 3] representing the frames.
            tensor2: Second tensor of shape [num_frames, num_joints, 3] representing the frames.
            nrows: Number of rows in the subplot grid (should be 2 for two tensors).
            ncols: Number of columns in the subplot grid (same for both tensors).
            title1: Title for the first tensor to be displayed at the top of the first row.
            title2: Title for the second tensor to be displayed at the top of the second row.
            out_dir: Output directory where the figure is stored.
            fname: File name for the saved figure.
        """
        num_frames1, num_joints, _ = tensor1.shape
        num_frames2, _, _ = tensor2.shape
        step_size1 = math.floor(num_frames1 / ncols)
        step_size2 = math.floor(num_frames2 / ncols)
        elev, azim = rotation
        parents = self.SMPL_PARENTS

        # Adjust the figure size based on the number of subplots
        fig_width = 1 * ncols  # Adjust the width as necessary
        fig_height = 2 * nrows  # Adjust the height as necessary
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": "3d"},
            figsize=(fig_width, fig_height),
        )

        # Display the first tensor in the first row
        for i in range(ncols):
            frame_index = i * step_size1
            if frame_index >= num_frames1:
                break
            frame = tensor1[frame_index]
            ax = axes[0, i] if ncols > 1 else axes[0]
            self.plot_skeletal_frame(
                ax,
                frame,
                parents,
                title1 if i == ncols // 2 else "",
                "blue",
                frame_index,
                elev,
                azim,
            )

        # Display the second tensor in the second row
        for i in range(ncols):
            frame_index = i * step_size2
            if frame_index >= num_frames2:
                break
            frame = tensor2[frame_index]
            ax = axes[1, i] if ncols > 1 else axes[1]
            self.plot_skeletal_frame(
                ax,
                frame,
                parents,
                title2 if i == ncols // 2 else "",
                "green",
                frame_index,
                elev,
                azim,
            )

        # Adjust layout to make subplots touch each other
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save or show the figure
        if out_dir and fname:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_path = os.path.join(out_dir, fname + ".png")
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

        plt.close(fig)

    def display_skeletal_frames(
        self,
        tensor,
        tensor_title,
        ncols=30,
        rotation=(0, -60),
        out_dir=None,
        fname=None,
    ):
        """
        Display skeletal frames from a tensor in a single row of subplots,
        connecting joints based on the parent body model. The axes are completely hidden for a clean look,
        and the tensor name is displayed at the top center. The view can be rotated based on the given
        elevation and azimuth angles.
        Args:
            tensor: Tensor of shape [num_frames, num_joints, 3] representing the frames.
            ncols: Number of columns in the subplot grid.
            tensor_title: Title for the tensor to be displayed at the top center.
            rotation: A tuple (elev, azim) for setting the view rotation.
            out_dir: Output directory where the figure is stored.
            fname: File name for the saved figure.
        """
        num_frames, num_joints, _ = tensor.shape
        nrows = 1  # Only one row for a single tensor
        step_size = math.floor(num_frames / ncols)
        parents = self.SMPL_PARENTS

        # Adjust the figure size based on the number of subplots
        fig_width = 4 * ncols  # Adjust the width as necessary
        fig_height = 3  # One row height
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": "3d"},
            figsize=(fig_width, fig_height),
        )

        # Flatten the axes array for easy indexing if there are multiple rows
        axes = axes.flatten() if ncols > 1 else [axes]

        for i in range(ncols):
            frame_index = i * step_size
            if frame_index >= num_frames:
                break
            frame = tensor[frame_index]
            ax = axes[i]
            self.plot_skeletal_frame(
                ax,
                frame,
                parents,
                tensor_title if i == ncols // 2 else "",
                "blue",
                frame_index,
                *rotation,
            )

        # Adjust layout to make subplots touch each other
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save or show the figure
        if out_dir and fname:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_path = os.path.join(out_dir, fname + ".png")
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

        plt.close(fig)

    def plot_skeletal_frame(
        self, ax, frame, parents, title, color, frame_index, elev, azim
    ):
        """
        Helper function to plot a single skeletal frame with frame index label.
        """
        num_joints = frame.shape[0]
        for j in range(num_joints):
            if parents[j] != -1:  # -1 indicates no parent
                joint = frame[j]
                parent_joint = frame[parents[j]]
                ax.plot(
                    [joint[0], parent_joint[0]],
                    [joint[1], parent_joint[1]],
                    [joint[2], parent_joint[2]],
                    marker="o",
                    markersize=1,
                    color=color,
                    linestyle="-",
                    linewidth=1,
                )
        ax.axis("off")

        # Set the figure title
        if title:
            ax.set_title(title, y=1.05)

        # Set the view angle
        ax.view_init(elev=elev, azim=azim)

        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1,1)

        # Add frame number below the subplot
        ax.text2D(
            0.5,
            -0.05,
            f"Frame {frame_index}",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
