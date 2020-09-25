# Final-Year-Project
The objective of the project is to solve the light field images novel view synthesis and baseline magnification problem:

Novel View Synthesis: Given a set of input images of an object viewing from different angles, and to synthesis an image from a synthetic view. The input can be stereo images/light field images/multi-images etc.

Light Field Images Input: The light field images can simply be seen as a collection of images captured by a set of equally spaced rgb camera, and in the problem abstraction, the camera is usually modeled as pin-hole camera.

The novel view synthesis strategy can be classified as two types: one is the end to end view synthesis method, one is the 3D information reconstruction method[1][2]. In comparison, the 3D information reconstruction method is more robust and is generally faster when doing inference. The 3D information reconstruction methodology can be further be classified into multi-plane representation based and point cloud based rendering method. Both of these two methods captures the depth information and explicitly encodes this information to the output
