"""
Extension to the pims library to read Phantom cine files
based on python-cine
https://github.com/cbenhagen/python-cine
"""

# Copy from pims/norpix_reader.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


from pims.frame import Frame
from pims.base_frames import FramesSequence, index_attr
import os, struct
import datetime
import numpy as np
import pytz

from . import cine
from . import linLUT

__all__ = ['CineSequence']

class CineSequence(FramesSequence):
    """Read Cine files produced by Phantom camera
    """

    @classmethod
    def class_exts(cls):
        return {'cine'} | super(CineSequence, cls).class_exts()

    # propagate_attrs = ['frame_shape', 'pixel_type', 'get_time',
    #                    'get_time_float', 'filename', 'width', 'height',
    #                    'frame_rate']

    # Other Bayer patterns not implemented
    bayerpatterns = {0: None, 3: 'gbrg', 4: 'rggb'}


    def __init__(self, filename, process_func=None, dtype=None,
                 calibrated = True, tzinfo=False):
        super(CineSequence, self).__init__()
        # tzinfo can be None if you want naive times,
        # or you can arbitrarily set it,
        # otherwise it takes the timezone from the header
        if tzinfo is not False:
            self.tzinfo = tzinfo
        self._verbose = False
        self._file = open(filename, 'rb')
        self._filename = filename
        self._calibrated = calibrated

        self._header = self._read_header()
        self._cinefileheader = self._header['cinefileheader']
        self._bitmapinfoheader = self._header['bitmapinfoheader']
        self._setup = self._header['setup']
        self._pImage = self._header['pImage']

        if 'frametime' in self._header:
            self._frametime = self._header['frametime']
        if 'frametime_float' in self._header:
            self._frametime_float = self._header['frametime_float']
        if 'exposure' in self._header:
            self._exposure = self._header['exposure']

        self._image_count = self._cinefileheader.ImageCount
        self._width = self._bitmapinfoheader.biWidth
        self._height = self._bitmapinfoheader.biHeight
        if self._bitmapinfoheader.biCompression:
            self._bpp = 12
        else:
            self._bpp = self._setup.RealBPP
        if self._bpp > 8:
            # Not really, but there's no np.uint12
            self._dtype_native = np.uint16
        else:
            self._dtype_native = np.uint8
        if self._calibrated:
            self._dtype = np.float32
        else:
            self._dtype = self._dtype_native

        # These keys are from Norpix.  There may be more that are expected
        self.metadata = {
            'width': self._width,
            'height':self._height,
            'bit_depth_real': self._bpp, # Real bit depth before compression
            # 'description': ''
            # 'origin': None, # How is this described?  This is not 'which corner is 0'
            'suggested_frame_rate': self._setup.FrameRate
        }


    def _read_header(self):
        with open(self._filename, 'rb') as f:
            f.seek(0)
            header = {}
            header['cinefileheader'] = cine.CINEFILEHEADER()
            header['bitmapinfoheader'] = cine.BITMAPINFOHEADER()
            header['setup'] = cine.SETUP()
            f.readinto(header['cinefileheader'])
            f.readinto(header['bitmapinfoheader'])
            f.readinto(header['setup'])
            imagecount = header['cinefileheader'].ImageCount

            if not hasattr(self, 'tzinfo'):
                self.tzinfo = pytz.UTC
            f.seek(header['cinefileheader'].OffSetup +
                   header['setup'].Length)
            endofsetup = f.tell()
            # Do we have tagged information blocks?
            tscale = 1.0/(2**32)
            while f.tell() < header['cinefileheader'].OffImageOffsets :
                blocksize, blocktype, _ = struct.unpack(
                    '<IHH',f.read(8))
                data = f.read(blocksize - 8)
                if blocktype == 1000:
                    # Analog and digital signals; no longer used
                    continue
                elif blocktype == 1001:
                    # Image time tagged block; no longer used
                    continue
                elif blocktype == 1002:
                    # Time only block
                    t64s = struct.unpack('{}q'.format(imagecount),
                                         data)

                    header['frametime_float'] = np.array([
                        t64*tscale for t64 in t64s])
                    # This is the slowest part of opening megaframe files,
                    # so don't convert from flow until requested
                    # header['frametime'] = [
                    #     datetime.datetime.fromtimestamp(ts,
                    #                                     tz=self.tzinfo)
                    #     for ts in header['frametime_float']]
                elif blocktype == 1003:
                    # Exposure time block
                    exposures = struct.unpack('{}I'.format(imagecount),
                                              data)
                    header['exposure'] = [exposure * tscale
                        for exposure in exposures]
                elif blocktype == 1007:
                    # Timecode block
                    # I don't read this, but it could be useful
                    continue
                else:
                    print("Uninterpreted block type {}".format(blocktype))

            # header_length = ctypes.sizeof(header['cinefileheader'])
            # bitmapinfo_length = ctypes.sizeof(header['bitmapinfoheader'])

            f.seek(header['cinefileheader'].OffImageOffsets)
            header['pImage'] = struct.unpack('{}q'.format(imagecount),
                                             f.read(imagecount * 8))
        return header

    def _verify_frame_no(self, i):
        if int(i) != i:
            raise ValueError("Frame numbers can only be integers")
        if i >= self._image_count or i < 0:
            raise ValueError("Frame number is out of range: " + str(i))

    def _timefromfloat(self, time_float):
        return datetime.datetime.fromtimestamp(
                    time_float, tz=self.tzinfo)

    def get_frame(self, i):
        self._verify_frame_no(i)
        self._file.seek(self._pImage[i])
        metadata = {'time':self.get_time(i),
                    'time_float':self.get_time_float(i)}
        try:
            metadata['exposure'] = self._exposure[i]
        except:
            pass
        annotationsize, = struct.unpack('<I', self._file.read(4))
        annotationdata = self._file.read(annotationsize-8)
        imagesize, = struct.unpack('<I', self._file.read(4))
        data = self._file.read(imagesize)
        rawimage = self._create_raw_array(data)
        if self._calibrated:
            rawimage = self._color_pipeline(rawimage)
            #raise NotImplementedError("Calibrated images not yet implemented")

        return Frame(rawimage, frame_no=i, metadata=metadata)


    def _create_raw_array(self, data):
        if self._bitmapinfoheader.biCompression:
            rawimage = self._unpack_10bit(data)
            self._fix_bad_pixels(rawimage)
            rawimage = linLUT.linLUT.astype(np.uint16)[rawimage]
            # rawimage = np.interp(rawimage, [64, 4064], [0, 2**12-1]).astype(np.uint16)
        else:
            rawimage = np.frombuffer(data, dtype='uint16')
            rawimage.shape = (self._height, self._width)
            self._fix_bad_pixels(rawimage)
            # Why is biCompression not flipped if this is?
            # rawimage = np.flipud(rawimage)
            # rawimage = np.interp(rawimage,
            #           [self._setup.BlackLevel, self._setup.WhiteLevel],
            #           [0, 2 ** self._setup.RealBPP - 1]).astype(np.uint16)

        if self._setup.bFlipH or self._setup.bFlipV:
            rawimage = rawimage[::-1 if self._setup.bFlipV else 1,
                                ::-1 if self._setup.bFlipH else 1]
        return rawimage



    def _unpack_10bit(self, data):
        packed = np.frombuffer(data, dtype='uint8').astype('uint16')
        unpacked = np.empty([self._height, self._width], dtype='uint16')

        unpacked.flat[::4] = (packed[::5] << 2) | (packed[1::5] >> 6)
        unpacked.flat[1::4] = ((packed[1::5] & 0b00111111) << 4) | (packed[2::5] >> 4)
        unpacked.flat[2::4] = ((packed[2::5] & 0b00001111) << 6) | (packed[3::5] >> 2)
        unpacked.flat[3::4] = ((packed[3::5] & 0b00000011) << 8) | packed[4::5]

        return unpacked


    def _fix_bad_pixels(self, rawimage):
        pattern = self.bayerpatterns[self._setup.CFA]
        whitelevel = self._setup.WhiteLevel
        hot = np.where(rawimage > whitelevel)
        coordinates = zip(hot[0], hot[1])

        if pattern is None:
            # Gray scale, use simpler algorithm
            # (median of non-hot pixels in 3x3 neighborhood)
            # Fails for hot regions larger than 3x3
            for coord in coordinates:
                localpix = rawimage[coord[0]-1:coord[0]+2,
                                    coord[1]-1:coord[1]+2].ravel()
                localpix = localpix[np.where(localpix <= whitelevel)]
                rawimage[coord[0],coord[1]] = np.median(localpix)
        else:
            masked_image = np.ma.MaskedArray(rawimage)

            for color in 'rgb':
                # FIXME: reuse those masks for whitebalancing
                mask = self._gen_mask(pattern, color, rawimage)
                masked_image.mask = mask
                smooth = cv2.medianBlur(masked_image, ksize=3)

                for coord in coordinates:
                    if not mask[coord]:
                        if self._verbose:
                            print('fixing {} in color {}'.format(coord, color))
                        rawimage[coord] = smooth[coord]
                if self._verbose:
                    print('done color', color)

            masked_image.mask = np.ma.nomask

    def _gen_mask(self, pattern, c, image):
        def color_kern(pattern, c):
            return np.array([[pattern[0] != c, pattern[1] != c],
                             [pattern[2] != c, pattern[3] != c]])

        (h, w) = image.shape[:2]
        cells = np.ones((h//2, w//2))

        return np.kron(cells, color_kern(pattern, c))

    def _color_pipeline(self, raw):
        """Order from:
        http://www.visionresearch.com/phantomzone/viewtopic.php?f=20&t=572#p3884
        """

        # For grayscale, use the fllowing pipeline
        setup = self._setup
        if 0 == setup.CFA:
            # convert to float
            image = raw.astype(np.float32) / (2**self._bpp-1)
            if setup.fOffset != 0:
                image += setup.fOffset
            if setup.fGain != 1.0:
                image *= setup.fGain
            image = self._apply_gamma(image)
            return image

        # 1. Offset the raw image by the amount in flare
        if self._verbose:
            print("fFlare: ", setup.fFlare)

        # 2. White balance the raw picture using the white balance component of cmatrix
        BayerPatterns = {3: 'gbrg', 4: 'rggb'}
        pattern = BayerPatterns[setup.CFA]

        self._whitebalance_raw(raw)

        # 3. Debayer the image
        rgb_image = cv2.cvtColor(raw, cv2.COLOR_BAYER_GB2RGB)

        # convert to float
        rgb_image = rgb_image.astype(np.float32) / (2**self._bpp-1)

        # 4. Apply the color correction matrix component of cmatrix
        """
        From the documentation:
        ...should decompose this
        matrix in two components: a diagonal one with the white balance to be
        applied before interpolation and a normalized one to be applied after
        interpolation.
        """
        cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)

        # normalize matrix
        ccm = cmCalib / cmCalib.sum(axis=1)[:, np.newaxis]

        # or should it be normalized this way?
        ccm2 = cmCalib.copy()
        ccm2[0][0] = 1 - ccm2[0][1] - ccm2[0][2]
        ccm2[1][1] = 1 - ccm2[1][0] - ccm2[1][2]
        ccm2[2][2] = 1 - ccm2[2][0] - ccm2[2][1]

        if self._verbose:
            print("cmCalib", cmCalib)
            print("ccm: ", ccm)
            print("ccm2", ccm2)

        rgb_image = np.dot(rgb_image, ccm.T)

        # 5. Apply the user RGB matrix umatrix
        cmUser = np.asarray(setup.cmUser).reshape(3, 3)
        rgb_image = np.dot(rgb_image, cmUser.T)

        # 6. Offset the image by the amount in offset
        # print("fOffset: ", setup.fOffset)

        # 7. Apply the global gain
        # print("fGain: ", setup.fGain)

        # 8. Apply the per-component gains red, green, blue
        # print("fGainR, fGainG, fGainB: ", setup.fGainR, setup.fGainG, setup.fGainB)

        # 9. Apply the gamma curves; the green channel uses gamma, red uses gamma + rgamma and blue uses gamma + bgamma
        # print("fGamma, fGammaR, fGammaB: ", setup.fGamma, setup.fGammaR, setup.fGammaB)
        rgb_image = self._apply_gamma(rgb_image)

        # 10. Apply the tone curve to each of the red, green, blue channels
        fTone = np.asarray(setup.fTone)
        # print(setup.ToneLabel, setup.TonePoints, fTone)

        # 11. Add the pedestals to each color channel, and linearly rescale to keep the white point the same.
        # print("fPedestalR, fPedestalG, fPedestalB: ", setup.fPedestalR, setup.fPedestalG, setup.fPedestalB)

        # 12. Convert to YCrCb using REC709 coefficients

        # 13. Scale the Cr and Cb components by chroma.
        # print("fChroma: ", setup.fChroma)

        # 14. Rotate the Cr and Cb components around the origin in the CrCb plane by hue degrees.
        # print("fHue: ", setup.fHue)

        return rgb_image

    def _apply_gamma(self, rgb_image):
        # FIXME: using 2.2 for now because 8.0 from the sample image seems way out of place
        rgb_image **= (1/2.2)
        # rgb_image **= (1/setup.fGamma)
        # rgb_image[:,:,0] **= (1/(setup.fGammaR + setup.fGamma))
        # rgb_image[:,:,2] **= (1/(setup.fGammaB + setup.fGamma))

        return rgb_image

    def _whitebalance_raw(self, raw):
        setup = self._setup
        pattern = self.bayerpatterns[setup.CFA]
        cmCalib = np.asarray(setup.cmCalib).reshape(3, 3)
        whitebalance = np.diag(cmCalib)
        if self._verbose:
            print("WBGain: ", np.asarray(setup.WBGain))
            print("WBView: ", np.asarray(setup.WBView))
            print("fWBTemp: ", setup.fWBTemp)
            print("fWBCc: ", setup.fWBCc)
            print("cmCalib: ", cmCalib)
            print("whitebalance: ", whitebalance)

        # FIXME: maybe use .copy()
        wb_raw = np.ma.MaskedArray(raw).astype(np.float16)

        wb_raw.mask = self._gen_mask(pattern, 'r', wb_raw)
        wb_raw *= whitebalance[0]
        wb_raw.mask = self._gen_mask(pattern, 'g', wb_raw)
        wb_raw *= whitebalance[1]
        wb_raw.mask = self._gen_mask(pattern, 'b', wb_raw)
        wb_raw *= whitebalance[2]

        wb_raw.mask = np.ma.nomask

        return wb_raw

    @index_attr
    def get_time(self, i):
        """Return the time of frame i as a datetime instance.

        """
        return datetime.datetime.fromtimestamp(
                self._frametime_float[i], tz=self.tzinfo)

    @index_attr
    def get_time_float(self, i):
        """Return the time of frame i as a floating-point number of seconds."""
        return self._frametime_float[i]

    def dump_times_float(self):
        """Return all frame times in file, as an array of floating-point numbers."""
        return self._frametime_float

    @property
    def filename(self):
        return self._filename

    @property
    def pixel_type(self):
        return self._dtype

    @property
    def width(self):
        return self.metadata['width']

    @property
    def height(self):
        return self.metadata['height']

    @property
    def frame_shape(self):
        return (self.metadata['height'], self.metadata['width'])

    @property
    def frame_rate(self):
        return self.metadata['suggested_frame_rate']

    def __len__(self):
        return self._image_count

    def close(self):
        self._file.close()

    def __repr__(self):
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {w}w x {h}h
Pixel Datatype: {dtype}""".format(filename=self.filename,
                                  count=len(self),
                                  h=self.frame_shape[0],
                                  w=self.frame_shape[1],
                                  dtype=self.pixel_type)


if __name__ == '__main__':
    import sys
    import cv2
    seqchart = CineSequence(os.path.dirname(__file__) + '/../testfiles/chart1.cine')
    imchart = seqchart[0]
    cv2.imshow("Chart", imchart/float(imchart.max()))
    cv2.waitKey(1)
    seq = CineSequence(sys.argv[1], calibrated=False)
    print(seq._filename)
    nshowframes = 5000
    imagetimes = seq.dump_times_float()
    cv2.namedWindow("Phantom")
    cv2.moveWindow("Phantom", 0,500)
    for iframe in range(0,len(imagetimes), len(imagetimes) // nshowframes):
        print("Frame ",iframe,"at", seq.get_time(iframe).isoformat())
        im = seq[iframe]
        cv2.imshow("Phantom", # seq.get_time(iframe).isoformat(),
                   im * 1/float(im.max()))
        cv2.waitKey(1)
    cv2.waitKey(0)
