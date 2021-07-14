import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

class disk_profile():
    """Class to perform different kinds of radial profiles for a disk.

    The main input is a fits image. The class has a number of methods:
      - slice(): performs an intensity profile along a slice of the image.
      - averaged_profile(): an azimuthally averaged radial profile of an
        axisymmetric source.
      - deprojected_image(): a deprojected image (radius vs azimuth) of an
        axisymmetric source.

    Attributes:
      im_name:
        Name of fits image.
      image:
        Array with original image.
      header:
        Full header of input image.
      cellsize:
        pixel size of input image.
      IntUnit:
        Units of pixels of input image.
      bmaj, bmin:
        major and minor axis of beam.
      cent:
        pixel coordinates of center used for the specific profile or
        deprojection.
      nx, ny:
        Number of pixels along x and y axis.
      inc, pa:
        inclination and PA used for the specific profile or deprojection, in
        arcsec.
      xrot, yrot:
        Rotated 2D arrays of x and y coordinates for each pixel. Computed with
        deprojected_grid() method.
      rrot:
        2D array of radial coordinate for each pixel, in arcsec. Computed with
        deprojected_grid() method.
      phi:
        2D array of azimuthal angle for each pixel, in degrees. Computed with
        deprojected_grid() method.
      radii:
        1D vector with radial coordinates of azimuthally averaged profile,
        in arcsec. Computed with averaged_profile().
      int_aver:
        1D vector with azimuthally averaged intensity profile. Computed with
        averaged_profile().
      int_aver_err:
        1D vector with uncertainty of azimuthally averaged intensity profile.
        Computed with averaged_profile().
      err_type:
        Type of uncertainty used for azimuthally averaged intensity profile.
      radii_slice:
        1D vector with radial coordinates of slice profile, in arcsec.
        Computed with slice().
      int_slice:
        1D vector with intensity of slice profile. Computed with slice().
      int_slice_err:
        1D vector with uncertainty of slice profile. Computed with slice().
      deproj_image:
        2D array with deprojected image (radius vs azimtuh). Computed with
        deprojected_image().
      deproj_radii:
        1D vector with radial coordinates of deproj_image. Computed with
        deprojected_image().
      deproj_phi:
        1D vector with azimuthal coordinates of deproj_image. Computed with
        deprojected_image().
    """
    def __init__(self, im_name):
        """Initializes disk_profile.

        Args:
          im_name: string with name if fits input image.
        """
        imObj = pyfits.open(im_name)
        self.im_name = im_name
        # We check the shape of the image
        if len(imObj[0].data.shape) == 4:
            self.image = imObj[0].data[0, 0, :, :]
        elif len(imObj[0].data.shape) == 3:
            self.image = imObj[0].data[0, :, :]
        elif len(imObj[0].data.shape) == 2:
            self.image = imObj[0].data[:, :]
        else:
            raise IOError('Incorrect shape of input image.')

        # We get some information from the header
        Header = imObj[0].header
        imObj.close()
        try:
            if Header['cunit1'] == 'deg':
                self.cellsize = abs(Header['cdelt1']) * 3600.
            elif Header['cunit1'] == 'rad':
                self.cellsize = abs(Header['cdelt1']) * 180. / np.pi * 3600.
        except:
            # If cunit1 is not provided, we assume that cdelt is given in deg
            print('WARNING: cunit not in header.'
                  ' Assuming cdelt is given in degrees.\n')
            self.cellsize = abs(Header['cdelt1']) * 3600. # Cellsize (")
        try:
            self.IntUnit = Header['bunit'] # Brightness pixel units of image
        except:
            print('WARNING: bunit not in header. Assuming Jy/pixel.\n')
            self.IntUnit = 'Jy/pixel'
        try:
            self.bmaj = Header['bmaj'] * 3600. # Beam major axis (")
            self.bmin = Header['bmin'] * 3600. # Beam minor axis (")
        except:
            print('WARNING: beam not provided in header. Using pixel size.\n')
            self.bmaj = self.cellsize
            self.bmin = self.cellsize

        self.cent = [Header['crpix1'], Header['crpix2']]

        self.nx = Header['naxis1'] # Number of pixels in the x axis
        self.ny = Header['naxis2'] # Number of pixels in the y axis

        # Initialize inc and pa
        self.inc = 0.0
        self.pa = 0.0
        self.header = Header

    def deprojected_grid(self, **kwargs):
        """Creates a deprojected grid.

        Kwargs:
          inc, pa:
            inclination and PA used for the specific profile or deprojection, in
            arcsec.
          cent:
            pixel coordinates of center used for the specific profile or
            deprojection.
        """
        if 'cent' in kwargs:
            self.cent = kwargs['cent']
        if 'inc' in kwargs:
            self.inc = kwargs['inc']
        if 'pa' in kwargs:
            self.pa = kwargs['pa']

        # We build arrays for the x and y positions in the image
        xarray = np.array([range(self.nx) for i in
            range(self.ny)]) - self.cent[0]
        yarray = np.array([range(self.ny) for i in
            range(self.nx)]).T - self.cent[1]

        # We rotate these arrays according to the PA of the source
        # The input angle should be a PA (from N to E), but we subtract 90
        # degrees for this rotation
        theta_pa = np.deg2rad(self.pa - 90.)
        self.xrot = xarray * np.cos(-theta_pa) - yarray * np.sin(-theta_pa)
        self.yrot = xarray * np.sin(-theta_pa) + yarray * np.cos(-theta_pa)

        # We rotate the y positions according to the inclination of the disk
        self.yrot = self.yrot / np.cos(np.deg2rad(self.inc))

        # Array with the radii
        self.rrot = np.sqrt(self.xrot**2 + self.yrot**2) * self.cellsize

        # Array with the azimuthal angles
        # (yarray and xarray ar used so that phi is the azimuthal angle in the
        # plane of the sky)
        self.phi = np.arctan2(yarray, xarray)
        # We make all the angles positive
        self.phi[np.where(self.phi<0.0)] += 2. * np.pi

    def averaged_profile(self, rmax = 1.0, rmin = 0.0, dr = None,
                         ring_width = None, phi_min=0.0, phi_max=2.*np.pi,
                         err_type='std_a', rms = np.nan, do_model = False,
                         **kwargs):
        """Computes azimuthally averaged intensity profile.

        Args:
          rmin:
            minimum radii for the image, in arcsec.
          rmax:
            maximum radii for the image, in arcsec.
          rms:
            rms of the map (in Jy/beam).
          dr:
            Optional; size of radial pixel of output deprojected image,
            in arcsec.
          ring_width:
            Optional; width of the rings, in arcsec. If not given as input, it
            will be set to one third of the beam size.
          phi_min, phi_max:
            Optional; Minimum and maximum azimuthal angles, in case particular
            angles want to be selected when averaging the emission. Defaults are
            0 and 2pi.
          err_type:
            Optional; it sets how the uncertainty of the profile is calculated.
            Default is 'rms'. Options are:
              - 'rms_a', it will use the rms of the image divided by the square
              root of the area of the ring (in beams).
              - 'rms_l', it will use the rms of the image divided by the square
              root of the length of the ring (in beams).
              - 'std', it will use the standard deviation inside each ring.
              - 'std_a', it will use the standard deviation inside each ring
              divided by the square root of the area of the ring (in beams).
              - 'std_l', it will use the standard deviation inside each ring
              divided by the square root of the length of the ring (in beams).
          do_model:
              Optional; if True, a "model" image will be created with the
              emission of each ring, together with an image of the uncertainty
              in each ring. Default is False.
        """
        if ('inc' in kwargs) or ('pa' in kwargs) or ('cent' in kwargs):
            self.deprojected_grid(**kwargs)
        else:
            try:
                rrot = self.rrot
            except AttributeError:
                self.deprojected_grid()

        if dr == None:
            dr = np.sqrt(self.bmaj * self.bmin) / 10. # 1/10th of the beam size
        if ring_width == None:
            ring_width = np.sqrt(self.bmaj * self.bmin) / 3. # 1/3 of the beam size

        # Vector of radii for the position of the annular rings, from rmin to rmax
        nr = (rmax - rmin) / dr # Number of radii
        radii = np.arange(int(nr) + 1) * (rmax - rmin)/nr + rmin + dr/2.

        Abeam = np.pi * self.bmaj * self.bmin/(4.0 * np.log(2.0)) # Area of the beam
        Lbeam = np.sqrt(self.bmaj * self.bmin) # "length" of the beam

        if err_type == 'rms_a' or err_type == 'rms_l':
            if rms == np.nan:
                print('WARNING: rms not provided. Uncertainties will be NaN')

        if do_model:
            model = np.ones_like(self.image)

        # For each ring, we will calculate its average intensity and uncertainty
        int_aver = []
        int_aver_err = []
        # We start averaging the emission
        for r in radii:
            r0 = r - ring_width / 2.
            r1 = r + ring_width / 2.
            if r0 < 0.0:
                r0 = 0.0
            Ring = self.image[(self.rrot >= r0) & (self.rrot < r1) &
            (self.phi >= phi_min) & (self.phi <= phi_max)]

            if err_type != 'percentiles':
                int_aver.append(np.nanmean(Ring))
            else:
                int_aver.append(np.nanmedian(Ring))

            if err_type == 'rms_a' or err_type == 'std_a':
                Aring = np.pi * np.cos(self.inc * np.pi/180.) * (r1**2 - r0**2) # Area of ring
                nbeams = Aring / Abeam # Number of beams in the ring
                if nbeams < 1.0:
                    nbeams = 1.0 # The error cannot be higher than one rms or std
                int_aver_err0 = 1.0/np.sqrt(nbeams)
                if err_type == 'std_a':
                    # If std_a, we multiply by the standard deviation inside the ring
                    int_aver_err0 *= np.nanstd(Ring)
                else:
                    int_aver_err0 *= rms
            elif err_type == 'rms_l' or err_type == 'std_l':
                a = r
                b = r * np.cos(self.inc * np.pi/180.)
                Lring = np.pi * (3. * (a + b) - np.sqrt((3.*a + b) *
                    (a + 3.*b))) # Length of elipse
                # NOTE: this will not be accurate for high inclinations and low nbeams
                nbeams = Lring / Lbeam
                if nbeams < 1.0:
                    nbeams = 1.0 # The error cannot be higher than one rms or std
                int_aver_err0 = 1.0/np.sqrt(nbeams)
                if err_type == 'std_l':
                    # If std_l, we multiply by the standard deviation inside the ring
                    int_aver_err0 *= np.nanstd(Ring)
                else:
                    int_aver_err0 *= rms
            elif err_type == 'std':
                int_aver_err0 = np.nanstd(Ring)
            elif err_type == 'percentiles':
                int_aver_err84 = np.nanpercentile(Ring,84)
                int_aver_err16 = np.nanpercentile(Ring,16)
                int_aver_err0 = [int_aver_err84, int_aver_err16]
            else:
                raise IOError('Wrong err_type: Type of uncertainty (err_type)'
                              ' is not recognised.')
            int_aver_err.append(int_aver_err0)

            if do_model:
                model[(self.rrot >= r0) & (self.rrot < r1)] = int_aver[-1]

        self.radii = np.array(radii)
        self.int_aver = np.array(int_aver)
        self.int_aver_err = np.array(int_aver_err)
        self.err_type = err_type

        if do_model:
            pyfits.writeto(
                self.im_name[:-5] + '.Model.fits', model, self.header,
                overwrite=True)

    def export_profile(self, outfile = None):
        """Exports azimuthally averaged intensity profile.

        Args:
          outfile:
            Name of the output files (.txt and .pdf). Default is the
            image name, without extension.
        """
        if outfile == None:
            outfile = self.im_name[:-5] + '.csv'
        f = open(outfile, 'w')
        if self.err_type == 'percentiles':
            f.write('Radius[arcsec],Int[{}],84_percentile,16_percentile\n'.format(self.IntUnit))
            for r, A, ErrA in zip(self.radii, self.int_aver, self.int_aver_err):
                f.write('{},{},{},{}\n'.format(r, A, ErrA[0], ErrA[1]))
            f.close()
        else:
            f.write('Radius[arcsec],Int[{}],Int_err\n'.format(self.IntUnit))
            for r, A, ErrA in zip(self.radii, self.int_aver, self.int_aver_err):
                f.write('{},{},{}\n'.format(r, A, ErrA))
            f.close()

    def slice(self, rmax = 1.0, dr = None, width = None, rms = np.nan,
              **kwargs):
        """Creates intensity profile along a slice.

        Args:
          rmax:
             maximum radii for the radial profile, in arcsec.
          rms:
             rms of the map (in Jy/beam).
          dr:
            Optional; separation between points, in arcsec.
          width:
            Optional; width of the slice, in arcsec.
        """
        if ('pa' in kwargs) or ('cent' in kwargs):
            self.deprojected_grid(**kwargs)
        else:
            try:
                rrot = self.rrot
            except AttributeError:
                self.deprojected_grid()

        if dr == None:
            dr = 2.0 * self.cellsize
        if width == None:
            width = np.sqrt(self.bmaj * self.bmin) / 3. # 1/3 of the beam size

        # Vector of radii for the position where we calculate the slice
        nr = rmax / dr # Number of radii
        radii = np.arange((int(nr) + 1)) * rmax/nr + dr/2.
        radii = np.concatenate((-1.*radii[::-1], radii))

        inten = []
        for r in radii:
            r0 = r - dr / 2.
            r1 = r + dr / 2.
            point = self.image[(self.xrot >= r0) & (self.xrot < r1) &
                (self.yrot >= -1.*width/2.) & (self.yrot <= width/2.)]
            inten.append(np.nanmean(point))

        self.radii_slice = np.array(radii)
        self.int_slice = np.array(inten)
        self.int_slice_err = np.ones_like(self.int_slice) * rms

    def export_slice(self, outfile = None):
        """Exports slice profile.

        Args:
          outfile:
            Optional; Name of the output files (.txt and .pdf). Default is the
            image name, without extension.
        """
        # We write the output result in a file
        if outfile == None:
            outfile = self.im_name[:-5] + '.csv'
        f = open(outfile, 'w')
        f.write('Radius[arcsec],Int[{}],Int_err\n'.format(self.IntUnit))
        for r, Int, err in zip(self.radii_slice, self.int_slice,
        self.int_slice_err):
            f.write('{},{},{}\n'.format(r, Int, err))
        f.close()

    def deprojected_image(self, rmax = 1.0, dr = None, nphi = 50, **kwargs):
        """Creates deprojected image.

        Args:
          rmax:
            Optional; maximum radii for the image, in arcsec.
          dr:
            Optional; size of radial pixel of output deprojected image, in arcsec.
          nphi:
            Optional; number of bins in azimtuh angle.
        """
        if ('inc' in kwargs) or ('pa' in kwargs) or ('cent' in kwargs):
            self.deprojected_grid(**kwargs)
        else:
            try:
                rrot = self.rrot
            except AttributeError:
                self.deprojected_grid()

        if dr == None:
            dr = 2.0 * self.cellsize

        nr = int(rmax / dr)
        dphi = 2. * np.pi / nphi
        self.deproj_image = np.ones(shape=(nr, nphi))
        self.deproj_radii = np.arange(0, rmax, dr) + dr/2.
        self.deproj_phi = np.arange(0, 2. * np.pi, dphi) + dphi/2.
        for i in range(nr):
            r0 = self.deproj_radii[i] - dr/2.
            r1 = self.deproj_radii[i] + dr/2.
            for j in range(nphi):
                phi0 = self.deproj_phi[j] - dphi/2.
                phi1 = self.deproj_phi[j] + dphi/2.
                self.deproj_image[i,j] = np.average(
                    self.image[(self.rrot >= r0) & (self.rrot < r1) &
                    (self.phi >= phi0) & (self.phi <= phi1)])


def deproject_image(im_name, inc, pa, rmax = 1.0, cent = None, dr = 0.02,
                    nphi = 50, doPlot = False, rms = None, contours = False,
                    levels = [], outfile = None):
    """Creates a deprojected image, in polar coordinates (azimuth vs radii).

    Returns arrays with radii, azimuthal angles, and deprojected image.

    Args:
      im_name:
        name of the image, in fits format.
      inc, pa:
        inclination and position angle of the source, in degrees.
      rmax:
        Optional; maximum radii for the image, in arcsec.
      cent:
        Optional; tuple with the position of the center of the source
        (in pixels)
      dr:
        Optional; size of radial pixel of output deprojected image, in arcsec.
      nphi:
        Optional; number of bins in azimtuh angle.
      doPlot:
        Optional; if True, an image will be created. Default is False.
      rms:
        Optional; rms of the map. If provided, it can be used to set
        the contour levels or set the vmin of the deprojected image. Default is
        None.
      contours:
        Optional; if True, the image will be done in filled contours.
        Default is False.
      levels:
        Optional; list. Only used if contours=True. Sets the levels of
        the contours.
      outfile:
        Optional; Name of the output files (.pdf). Default is the image name,
        without extension
    Returns:
      disk_profile object.
    """
    deproj_img = disk_profile(im_name)
    if cent == None:
        deproj_img.deprojected_grid(inc = inc, pa = pa)
    else:
        deproj_img.deprojected_grid(inc = inc, pa = pa, cent = cent)

    deproj_img.deprojected_image(rmax = rmax, dr = dr, nphi = nphi)

    if doPlot:
        f = plt.figure()
        ax = f.add_subplot(111)
        if contours:
            if len(levels) == 0:
                levels = list(map(lambda x: x*np.max(deproj_img.deproj_image),
                    [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]))
            elif rms != None:
                levels = list(map(lambda x: x*rms,levels))
            else:
                raise IOError('rms needs to be provided if contour '+
                              'levels are provided.')
            ax.contourf(
                deproj_img.deproj_radii,
                deproj_img.deproj_phi / np.pi * 180.,
                deproj_img.deproj_image.T[::-1,:],
                aspect = rmax / 360., levels = levels)
        else:
            if rms != None:
                ax.imshow(
                    deproj_img.deproj_image.T, extent = (0.0,rmax,0.,360.),
                    aspect = rmax/360.*0.5, cmap = cm.viridis, vmin = rms,
                    vmax = np.max(deproj_img.deproj_image))
            else:
                ax.imshow(
                    deproj_img.deproj_image.T, extent=(0.0, rmax, 0., 360.),
                    aspect = rmax/360.*0.5, cmap = cm.viridis, vmin= 0.0,
                    vmax = np.max(deproj_img.deproj_image))
        ax.set_ylabel('Azimuth (deg)',fontsize=15)
        ax.set_xlabel('Radii (au)',fontsize=15)
        if outfile == None:
            outfile = im_name[:-5] + '_deproj.pdf'
        plt.savefig(outfile)
        plt.close(f)
    return deproj_img


def rad_slice(im_name, pa, rmax, rms=np.nan, cent=None, dr=None, width=None,
              fold=True, outfile=None, do_plot=True, color='#809fff', dist=None,
              ylim=None, ylog=False):
    """Creates a slice across the major axis of a source, from -rmax to rmax.

    Returns arrays with radii, integrated intensity and uncertainty.

    Args:
      im_name:
        name of the image, in fits format.
      pa:
        position angle of the source, in degrees.
      rmax:
         maximum radii for the radial profile, in arcsec.
      rms:
         rms of the map (in Jy/beam).
      cent:
        Optional; tuple with the position of the center of the source
        (in pixels)
      dr:
        Optional; separation between points, in arcsec.
      width:
        Optional; width of the slice, in arcsec.
      fold:
        Optional; if True, it will "fold" the slice along the center.
        Default is True.
      outfile:
        Optional; Name of the output files (.txt and .pdf). Default is the
        image name, without extension.
      do_plot:
        Optional; if True, a plot will be created with the profile.
      color:
        Optional; color of the plot. Default is blue.
      dist:
        Optional; distance to the source, in pc. It will only be necessary if
        a twin x axis with the distances in au is wanted.
      ylim:
        Optional; limit of y axis, in units of the plot. If not set, they
        will be selected automatically.
      ylog:
        Optional; if True, it will set the y scale to log. Default is False.
    Returns:
        disk_profile object.
    """
    slice_img = disk_profile(im_name)
    if cent == None:
        slice_img.deprojected_grid(pa = pa)
    else:
        slice_img.deprojected_grid(pa = pa, cent = cent)

    slice_img.slice(rmax = rmax, dr = dr, width = width, rms = rms)
    slice_img.export_slice(outfile = outfile)

    if fold:
        nr = len(slice_img.radii_slice)
        slice_prof = (slice_img.int_slice[int(nr/2):] +
            slice_img.int_slice[int(nr/2)-1::-1]) / 2.
        slice_rad = slice_img.radii_slice[int(nr/2):]
        slice_rad_err = np.sqrt((slice_prof -
            slice_img.int_slice[int(nr/2):])**2. + (slice_prof -
            slice_img.int_slice[int(nr/2)-1::-1])**2.) / np.sqrt(2.)
    else:
        slice_prof = slice_img.int_slice
        slice_rad = slice_img.radii_slice
        slice_rad_err = slice_img.int_slice_err

    # We make a plot with the profile
    if do_plot:
        # If the units of the image are Jy/beam, we make the plot in mJy/beam
        if slice_img.IntUnit == 'Jy/beam':
            ytitle = 'Average Intensity [mJy/beam]'
            slice_prof = slice_prof * 1000.0 #mJy/beam
            slice_rad_err = slice_rad_err * 1000.0
        else:
            ytitle = 'Average Intensity ['+slice_img.IntUnit+']'

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylabel(ytitle, fontsize=15)
        ax1.set_xlabel('Radius [arcsec]',fontsize=15)
        if fold:
            ax1.set_xlim([0.0, rmax])
        else:
            ax1.set_xlim([-rmax, rmax])
        #ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax1.tick_params(axis='both',direction='inout',which='both')

        if color == None:
            color = '#ffd480'
        if np.all(np.isnan(slice_rad_err)) == False:
            ax1.fill_between(
                slice_rad, slice_prof + slice_rad_err,
                slice_prof - slice_rad_err, facecolor = color)
        ax1.plot(slice_rad, slice_prof, 'k-')

        ax1.axhline(y=0.0, color='k', linestyle='--', lw=0.5)
        ax1.axvline(x=0.0, color='k', linestyle='--', lw=0.5)
        if ylim != None:
            ax1.set_ylim(ylim)

        if dist != None:
            twax1 = ax1.twiny()
            if fold:
                twax1.set_xlim([0.0, rmax * dist])
                twax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
                twax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
            else:
                twax1.set_xlim([-rmax * dist, rmax * dist])
                twax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))
                twax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
            twax1.set_xlabel('Radius [au]',fontsize=15)
            twax1.tick_params(direction='inout',which='both')

        if ylog:
            ax1.set_yscale('log')
        twax2 = ax1.twinx()
        twax2.set_ylim(ax1.get_ylim())
        #twax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        twax2.tick_params(labelright='off',direction='in',which='both')

        if outfile == None:
            outfile = im_name[:-5] + '_slice.pdf'
        plt.savefig(outfile, dpi = 650)
        plt.close(fig)

    return slice_img


def rad_profile(im_name, inc, pa, rmin, rmax, rms = np.nan, dr = None,
                cent = None, ring_width = None, phi_min = 0.0,
                phi_max = 2.*np.pi, err_type = 'rms_a', do_model = False,
                outfile = None, do_plot=True, color='#809fff', dist='',
                ylim=None, ylog=False):
    """Creates an azimuthally averaged radial profile.

   Returns arrays with radii, integrated intensity and uncertainty.

    Args:
      im_name:
        name of the image, in fits format.
      inc, pa:
        inclination and position angle of the source, in degrees.
      rmin:
        minimum radii for the image, in arcsec.
      rmax:
        maximum radii for the image, in arcsec.
      rms:
         rms of the map (in Jy/beam).
      cent:
        Optional; tuple with the position of the center of the source
        (in pixels). i.e. where the annular eliptical rings will be centered.
      dr:
        Optional; size of radial pixel of output deprojected image, in arcsec.
      ring_width:
        Optional; width of the rings, in arcsec. If not given as input, it will
        be set to one third of the beam size.
      phi_min, phi_max:
        Optional; Minimum and maximum azimuthal angles, in case particular
        angles want to be selected when averaging the emission. Defaults are
        0 and 2pi.
      err_type:
        Optional; it sets how the uncertainty of the profile is calculated.
        Default is 'rms'. Options are:
          - 'rms_a', it will use the rms of the image divided by the square root
          of the area of the ring (in beams).
          - 'rms_l', it will use the rms of the image divided by the square root
          of the length of the ring (in beams).
          - 'std', it will use the standard deviation inside each ring.
          - 'std_a', it will use the standard deviation inside each ring divided
          by the square root of the area of the ring (in beams).
          - 'std_l', it will use the standard deviation inside each ring divided
          by the square root of the length of the ring (in beams).
      do_model:
        Optional; if True, a "model" image will be created with the emission of
        each ring, together with an image of the uncertainty in each ring.
        Default is False.
      outfile:
        Optional; Name of the output files (.txt and .pdf). Default is the
        image name, without extension.
      do_plot:
        Optional; if True, a plot will be created with the profile.
      color:
        Optional; color of the plot. Default is blue.
      dist:
        Optional; distance to the source, in pc. It will only be necessary if
        a twin x axis with the distances in au is wanted.
      ylim:
        Optional; limit of y axis, in units of the plot. If not set, they
        will be selected automatically.
      ylog:
        Optional; if True, it will set the y scale to log. Default is False.
    Returns:
        disk_profile object.
    """
    profile_img = disk_profile(im_name)
    if cent == None:
        profile_img.deprojected_grid(inc = inc, pa = pa)
    else:
        profile_img.deprojected_grid(inc = inc, pa = pa, cent = cent)

    profile_img.averaged_profile(rmax = rmax, rmin = rmin, dr = dr,
    ring_width = ring_width, phi_min = phi_min, phi_max = phi_max,
    err_type = err_type, rms = rms, do_model = do_model)

    profile_img.export_profile(outfile = outfile)

    # We make a plot with the profile
    if do_plot:
        radii = profile_img.radii
        # If the units of the image are Jy/beam, we will make the plot in mJy/beam
        if profile_img.IntUnit == 'Jy/beam':
            ytitle = 'Average Intensity [mJy/beam]'
            int_aver = profile_img.int_aver * 1000.0 #mJy/beam
            int_aver_err = profile_img.int_aver_err * 1000.0 #mJy/beam
        else:
            ytitle = 'Average Intensity ['+profile_img.IntUnit+']'
            int_aver = profile_img.int_aver
            int_aver_err = profile_img.int_aver_err

        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylabel(ytitle,fontsize=15)
        ax1.set_xlabel('Radius [arcsec]',fontsize=15)
        ax1.set_xlim([0.0,rmax])
        #ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax1.tick_params(axis='both',direction='inout',which='both')
        if err_type == 'percentiles':
            ax1.fill_between(
                radii, int_aver_err[:,0], int_aver_err[:,1], facecolor=color)
        else:
            ax1.fill_between(
                radii, int_aver + int_aver_err, int_aver - int_aver_err,
                facecolor=color)
        ax1.plot(radii, int_aver, 'k-')

        ax1.axhline(y=0.0, color='k', linestyle='--', lw=0.5)
        if ylim != None:
            ax1.set_ylim(ylim)

        if dist != '':
            twax1 = ax1.twiny()
            twax1.set_xlim([0.0, rmax * dist])
            twax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            twax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
            twax1.set_xlabel('Radius [au]',fontsize=15)
            twax1.tick_params(direction='inout',which='both')

        if ylog:
            ax1.set_yscale('log')
        twax2 = ax1.twinx()
        twax2.set_ylim(ax1.get_ylim())
        #twax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        twax2.tick_params(labelright='off',direction='in',which='both')

        plt.savefig(outfile + '_profile.pdf',dpi = 650)
        plt.close(fig)

    return profile_img
