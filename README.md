# disk_profiles

Python module to obtain different kinds of intensity profiles from a
FITS image. Currently, three kinds of profiles can be obtained:
- slice profile: Profile along a slice in an image.
- azimuthally averaged profile: profile obtained by averaging the emission in
concentric rings, which can be deprojected with a certain inclination and PA.
- deprojected image: not a profile per se, but a deprojected image of azimuth
vs radius.
- azimuthal profile: azimuthal profile (deprojected or not), averaged within 
a given range of radii.
