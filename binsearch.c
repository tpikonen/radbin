
// binary search of index of val in monotonically increasing array table
// returns the index of the table or
// -1 if val < table[0]
// -2 if val > table[len-1]
int binsearch(double val, const double *table, int len)
{
    int left = 0;
    int right = len;
    int mid;
    while((right - left) > 1) {
	mid = (left + right) >> 1;
	if(val >= table[mid]) {
	    left = mid;
	} else {
	    right = mid;
	}
    }

    return left;
}

