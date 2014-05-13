
#include <opencv2/core/core_c.h>

#include "phough.h"

/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

namespace artelab
{

    const int STORAGE_SIZE = 1 << 12;

    void seqToMat(const CvSeq* seq, cv::OutputArray _arr)
    {
        if( seq && seq->total > 0 )
        {
            _arr.create(1, seq->total, seq->flags, -1, true);
            cv::Mat arr = _arr.getMat();
            cvCvtSeqToArray(seq, arr.data);
        }
        else
            _arr.release();
    }

    static void
    hough_prob(cv::Mat image, cv::Mat& accum, float rho, float theta, int threshold, int lineLength, int lineGap, CvSeq *lines, int linesMax)
    {
        cv::Mat mask;
        cv::vector<float> trigtab;
        cv::MemStorage storage(cvCreateMemStorage(0));

        CvSeq* seq;
        CvSeqWriter writer;
        int width, height;
        int numangle, numrho;
        float ang;
        int r, n, count;
        CvPoint pt;
        float irho = 1 / rho;
        CvRNG rng = cvRNG(-1);
        const float* ttab;
        uchar* mdata0;

        CV_Assert( image.depth() == CV_8U );

        width = image.cols;
        height = image.rows;

        numangle = cvRound(CV_PI / theta);
        numrho = cvRound(((width + height) * 2 + 1) / rho);

        accum.create( numangle, numrho, CV_32SC1 );
        mask.create( height, width, CV_8UC1 );
        trigtab.resize(numangle*2);
        accum = cv::Scalar(0);

        for( ang = 0, n = 0; n < numangle; ang += theta, n++ )
        {
            trigtab[n*2] = (float)(cos(ang) * irho);
            trigtab[n*2+1] = (float)(sin(ang) * irho);
        }
        ttab = &trigtab[0];
        mdata0 = mask.data;

        cvStartWriteSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage, &writer );

        // stage 1. collect non-zero image points
        for( pt.y = 0, count = 0; pt.y < height; pt.y++ )
        {
            const uchar* data = image.data + pt.y*image.step;
            uchar* mdata = mdata0 + pt.y*width;
            for( pt.x = 0; pt.x < width; pt.x++ )
            {
                if( data[pt.x] )
                {
                    mdata[pt.x] = (uchar)1;
                    CV_WRITE_SEQ_ELEM( pt, writer );
                }
                else
                    mdata[pt.x] = 0;
            }
        }

        seq = cvEndWriteSeq( &writer );
        count = seq->total;

        // stage 2. process all the points in random order
        for( ; count > 0; count-- )
        {
            // choose random point out of the remaining ones
            int idx = cvRandInt(&rng) % count;
            int max_val = threshold-1, max_n = 0;
            CvPoint* point = (CvPoint*)cvGetSeqElem( seq, idx );
            CvPoint line_end[2] = {{0,0}, {0,0}};
            float a, b;
            int* adata = (int*)accum.data;
            int i, j, k, x0, y0, dx0, dy0, xflag;
            int good_line;
            const int shift = 16;

            i = point->y;
            j = point->x;

            // "remove" it by overriding it with the last element
            *point = *(CvPoint*)cvGetSeqElem( seq, count-1 );

            // check if it has been excluded already (i.e. belongs to some other line)
            if( !mdata0[i*width + j] )
                continue;

            // update accumulator, find the most probable line
            for( n = 0; n < numangle; n++, adata += numrho )
            {
                r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
                r += (numrho - 1) / 2;
                int val = ++adata[r];
                if( max_val < val )
                {
                    max_val = val;
                    max_n = n;
                }
            }

            // if it is too "weak" candidate, continue with another point
            if( max_val < threshold )
                continue;

            // from the current point walk in each direction
            // along the found line and extract the line segment
            a = -ttab[max_n*2+1];
            b = ttab[max_n*2];
            x0 = j;
            y0 = i;
            if( fabs(a) > fabs(b) )
            {
                xflag = 1;
                dx0 = a > 0 ? 1 : -1;
                dy0 = cvRound( b*(1 << shift)/fabs(a) );
                y0 = (y0 << shift) + (1 << (shift-1));
            }
            else
            {
                xflag = 0;
                dy0 = b > 0 ? 1 : -1;
                dx0 = cvRound( a*(1 << shift)/fabs(b) );
                x0 = (x0 << shift) + (1 << (shift-1));
            }

            for( k = 0; k < 2; k++ )
            {
                int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

                if( k > 0 )
                    dx = -dx, dy = -dy;

                // walk along the line using fixed-point arithmetics,
                // stop at the image border or in case of too big gap
                for( ;; x += dx, y += dy )
                {
                    uchar* mdata;
                    int i1, j1;

                    if( xflag )
                    {
                        j1 = x;
                        i1 = y >> shift;
                    }
                    else
                    {
                        j1 = x >> shift;
                        i1 = y;
                    }

                    if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                        break;

                    mdata = mdata0 + i1*width + j1;

                    // for each non-zero point:
                    //    update line end,
                    //    clear the mask element
                    //    reset the gap
                    if( *mdata )
                    {
                        gap = 0;
                        line_end[k].y = i1;
                        line_end[k].x = j1;
                    }
                    else if( ++gap > lineGap )
                        break;
                }
            }

            good_line = abs(line_end[1].x - line_end[0].x) >= lineLength ||
                        abs(line_end[1].y - line_end[0].y) >= lineLength;

            for( k = 0; k < 2; k++ )
            {
                int x = x0, y = y0, dx = dx0, dy = dy0;

                if( k > 0 )
                    dx = -dx, dy = -dy;

                // walk along the line using fixed-point arithmetics,
                // stop at the image border or in case of too big gap
                for( ;; x += dx, y += dy )
                {
                    uchar* mdata;
                    int i1, j1;

                    if( xflag )
                    {
                        j1 = x;
                        i1 = y >> shift;
                    }
                    else
                    {
                        j1 = x >> shift;
                        i1 = y;
                    }

                    mdata = mdata0 + i1*width + j1;

                    // for each non-zero point:
                    //    update line end,
                    //    clear the mask element
                    //    reset the gap
                    if( *mdata )
                    {
                        if( good_line )
                        {
                            adata = (int*)accum.data;
                            for( n = 0; n < numangle; n++, adata += numrho )
                            {
                                r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                                r += (numrho - 1) / 2;
                                adata[r]--;
                            }
                        }
                        *mdata = 0;
                    }

                    if( i1 == line_end[k].y && j1 == line_end[k].x )
                        break;
                }
            }

            if( good_line )
            {
                CvRect lr = { line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y };
                cvSeqPush( lines, &lr );
                if( lines->total >= linesMax )
                    return;
            }
        }
    }

    void houghP( cv::Mat image, cv::OutputArray lines, cv::Mat& accumulator,
                          float rho, float theta, int threshold,
                          double minLineLength, double maxGap )
    {
        if( image.depth() != CV_8U)
            CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

        if( rho <= 0 || theta <= 0 || threshold <= 0 )
            CV_Error( CV_StsOutOfRange, "rho, theta and threshold must be positive" );

        int lineType = CV_32SC4;
        int elemSize = sizeof(int)*4;
        CvMemStorage* lineStorage = cvCreateMemStorage(STORAGE_SIZE);
        CvSeq* seq = cvCreateSeq(lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage);

        hough_prob(image, accumulator, rho, theta, threshold, minLineLength, maxGap, seq, INT_MAX);
        seqToMat(seq, lines);
    }

}