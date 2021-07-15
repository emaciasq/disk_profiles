import disk_profiles

im_name = 'TWHya_B6_cont.fits'
inc = 5.0
pa = 152.0
profile = disk_profiles.rad_profile(im_name, inc, pa, rmin = 0.0, rmax = 1.0,
                                    rms = 9.8e-6, outfile = 'test', dist=60.0,
                                    err_type = 'std')

slice = disk_profiles.rad_slice(im_name, pa, rms = 9.8e-6,
                                outfile = 'test_slice', dist = 60.0)

deproj_image = disk_profiles.deproject_image(im_name, inc, pa, do_plot = True,
                                             outfile = 'test_image')