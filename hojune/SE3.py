import numpy as np
import cv2

# Finds 3d coords of point in reference frame B from two z=1 plane projections
def match2p3d(R,t, v2A, v2B):
    PDash = np.zeros((3,4))
    PDash[0:3, 0:3] = R # se3AfromB.rotation
    PDash[0:3, 3] = t.ravel() # se3AfromB.translation
    A = np.array([
    [-1.0, 0.0, v2B[0], 0.0],
    [0.0, -1.0, v2B[1], 0.0],
    list(v2A[0] * PDash[2] - PDash[0]),
    list(v2A[1] * PDash[2] - PDash[0])])
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    vSmall = vh[3]
    if vSmall[3] == 0.0:
        vSmall[3] = 0.00001
    p3d = list(np.array(vSmall)/vSmall[3])
    return p3d


# def CalibImage::GuessInitialPose(ATANCamera &Camera):
#   # First, find a homography which maps the grid to the unprojected image coords
#   # Use the standard null-space-of-SVD-thing to find 9 homography parms
#   # (c.f. appendix of thesis)
  
#   int nPoints = mvGridCorners.size();
#   Matrix<> m2Nx9(2*nPoints, 9);
#   for(int n=0; n<nPoints; n++)
#     {
#       // First, un-project the points to the image plane
#       Vector<2> v2UnProj = Camera.UnProject(mvGridCorners[n].Params.v2Pos);
#       double u = v2UnProj[0];
#       double v = v2UnProj[1];
#       // Then fill in the matrix..
#       double x = mvGridCorners[n].irGridPos.x;
#       double y = mvGridCorners[n].irGridPos.y;
      
#       m2Nx9[n*2+0][0] = x;
#       m2Nx9[n*2+0][1] = y;
#       m2Nx9[n*2+0][2] = 1;
#       m2Nx9[n*2+0][3] = 0;
#       m2Nx9[n*2+0][4] = 0;
#       m2Nx9[n*2+0][5] = 0;
#       m2Nx9[n*2+0][6] = -x*u;
#       m2Nx9[n*2+0][7] = -y*u;
#       m2Nx9[n*2+0][8] = -u;

#       m2Nx9[n*2+1][0] = 0;
#       m2Nx9[n*2+1][1] = 0;
#       m2Nx9[n*2+1][2] = 0;
#       m2Nx9[n*2+1][3] = x;
#       m2Nx9[n*2+1][4] = y;
#       m2Nx9[n*2+1][5] = 1;
#       m2Nx9[n*2+1][6] = -x*v;
#       m2Nx9[n*2+1][7] = -y*v;
#       m2Nx9[n*2+1][8] = -v;
#     }

#   // The right null-space (should only be one) of the matrix gives the homography...
#   SVD<> svdHomography(m2Nx9);
#   Vector<9> vH = svdHomography.get_VT()[8];
#   Matrix<3> m3Homography;
#   m3Homography[0] = vH.slice<0,3>();
#   m3Homography[1] = vH.slice<3,3>();
#   m3Homography[2] = vH.slice<6,3>();
  
  
#   // Fix up possibly poorly conditioned bits of the homography
#   {
#     SVD<2> svdTopLeftBit(m3Homography.slice<0,0,2,2>());
#     Vector<2> v2Diagonal = svdTopLeftBit.get_diagonal();
#     m3Homography = m3Homography / v2Diagonal[0];
#     v2Diagonal = v2Diagonal / v2Diagonal[0];
#     double dLambda2 = v2Diagonal[1];
    
#     Vector<2> v2b;   // This is one hypothesis for v2b ; the other is the negative.
#     v2b[0] = 0.0;
#     v2b[1] = sqrt( 1.0 - (dLambda2 * dLambda2)); 
    
#     Vector<2> v2aprime = v2b * svdTopLeftBit.get_VT();
    
#     Vector<2> v2a = m3Homography[2].slice<0,2>();
#     double dDotProd = v2a * v2aprime;
    
#     if(dDotProd>0) 
#       m3Homography[2].slice<0,2>() = v2aprime;
#     else
#       m3Homography[2].slice<0,2>() = -v2aprime;
#   }
 
  
#   // OK, now turn homography into something 3D ...simple gram-schmidt ortho-norm
#   // Take 3x3 matrix H with column: abt
#   // And add a new 3rd column: abct
#   Matrix<3> mRotation;
#   Vector<3> vTranslation;
#   double dMag1 = sqrt(m3Homography.T()[0] * m3Homography.T()[0]);
#   m3Homography = m3Homography / dMag1;
  
#   mRotation.T()[0] = m3Homography.T()[0];
  
#   // ( all components of the first vector are removed from the second...
  
#   mRotation.T()[1] = m3Homography.T()[1] - m3Homography.T()[0]*(m3Homography.T()[0]*m3Homography.T()[1]); 
#   mRotation.T()[1] /= sqrt(mRotation.T()[1] * mRotation.T()[1]);
#   mRotation.T()[2] = mRotation.T()[0]^mRotation.T()[1];
#   vTranslation = m3Homography.T()[2];
  
#   // Store result
#   mse3CamFromWorld.get_rotation()=mRotation;
#   mse3CamFromWorld.get_translation() = vTranslation;