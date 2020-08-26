# Data Structure (Advanced)
## Bitmagic

## Recursion
###### Josephus problem
    int josephus(int n, int k)
    {
       if(n==1)return 1;
       return (josephus(n-1,k)+k-1)%n+1;
    }
###### Power Of A Number
    long long power(int N,int R)
    {
       if(R==1)
        return N;

        return ((N%1000000007)*(power(N,R-1)%1000000007))%1000000007;

    }
######  Possible Words From Phone Digits
    const string CHARS[] = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    void words(const string chars[], int a[], string word, int N, int depth);
    void possibleWords(int a[],int N)
    {
        //Your code here
        string word = "";
        words(CHARS, a, word, N, 0);
    }

    void words(const string chars[], int a[], string word, int N, int depth)
    {
        if(N == depth)
        {
            cout << word << ' ';
            return;
        }

        for(int i = 0; i < chars[a[depth]].length(); ++i)
        {
            words(chars, a, word + chars[a[depth]][i], N, depth + 1);
        }
    }
######  Lucky Numbers
    int g=2;
    bool isLucky(int n, int &counter) {         
            if(counter>n)
                return true;
            if(n%counter==0)
                return false;
            n=n-n/counter;counter++;
            isLucky(n,counter);
    }
## Array
###### Max Circular Subarray Sum
    int kadane(int arr[], int num){

        int m=arr[0],ans=arr[0];
        for(int i=1;i<num;i++)
        {
            m=max(m+arr[i],arr[i]);
            ans=max(ans,m);
        }
        return ans;
    }
    int circularSubarraySum(int arr[], int num){
         int x=kadane(arr,num);
        if(x<0)
        return x;
        int i,sum=0;
        for(i=0;i<num;i++)
        {
            sum+=arr[i];
            arr[i]=-arr[i];
        }
        int y=kadane(arr,num);
        return max(x,y+sum);

    }

###### Longest Subarray Of Evens And Odds
    int maxEvenOdd(int arr[], int n) 
    { 
       int res=1;
       int curr=1;
       for(int i=1;i<n;i++)
       {
           if((arr[i]%2==0&&arr[i-1]%2!=0)||(arr[i]%2!=0&&arr[i-1]%2==0))
           {
               curr++;
               res= max(res,curr);
           }
           else
           curr=1;
       }
       return res;
    }
###### Kadane's Algorithm
    int maxSubarraySum(int arr[], int n){
                int i;
                for(i=1;i<n;i++)
                {   if(arr[i-1]+arr[i]>arr[i])
                        arr[i]=arr[i-1]+arr[i];
                }
                int max=arr[0];
                for(i=1;i<n;i++)
                {   if(arr[i]>max)
                        max=arr[i];
                }
                if(max<0)
                    return -1;
                else return max;
    }
###### Check if array is sorted and rotated
    bool checkRotatedAndSorted(int arr[], int num){

        // Your code here
        int key1=-1,key2=-1,i,j;
        for(i=1;i<num;i++)
        {
            if(arr[i-1]<arr[i])
            continue;
            else
            {
                key1++;
            }
        }
        for(i=1;i<num;i++)
        {
            if(arr[i-1]>arr[i])
            continue;
            else
            {
                key2++;
            }
        }
        if(key1==0&&arr[num-1]<arr[0])
        return 1;
        if(key2==0&&arr[num-1]>arr[0])
        return 1;
        return 0;
    }
###### Stock buy and sell
    void stockBuySell(int a[], int n)
    {  // Your code here
        std::vector<int> v ;
            int j=0;
            for(int i=0;i<n-1;i++){
                if(a[i]<a[i+1]){
                    v.push_back(i);
                    j=i;
                    while(a[j]<=a[j+1]&& j<n-1){
                        j++;
                    }
                    v.push_back(j);
                    i=j;
                }
            }
            if(v.size()==0){
                cout<<"No Profit";
            }
            else{
            for(int i=0;i<v.size();i=i+2){
                cout<<'('<<v[i]<<' '<<v[i+1]<<')'<<' ';
            }
            }
    }
###### Trapping Rain Water
    int trappingWater(int arr[], int n){

        // Your code here
        int l[n]; 
        int r[n]; 
        int wat = 0; 
        l[0] = arr[0]; 
        for(int i = 1; i < n; i++) 
            l[i] = max(l[i - 1], arr[i]); 

        r[n - 1] = arr[n - 1]; 
        for (int i = n - 2; i >= 0; i--) 
            r[i] = max(r[i + 1], arr[i]); 
        for (int i = 0; i < n; i++) 
            wat += min(l[i], r[i]) - arr[i]; 

        return wat; 

    }
###### Maximum Index
    int maxIndexDiff(int arr[], int n) 
    { 
        int LMin[n];
        int RMax[n];  
        LMin[0] = arr[0];
        for( int i=1;i< n; i++){
            LMin[i] = min(arr[i],LMin[i-1]);
        } 
        RMax[n-1]= arr[n-1];
        for(int i=n-2; i>=0;i--){
            RMax[i]= max( arr[i], RMax[i+1]);
        }
        int i=0,j=0, maxDiff=-1;
        while( i<n && j < n){
            if(LMin[i] <= RMax[j]){
                maxDiff= max(maxDiff,j-i);
                j++;
            }
            else 
              i++;
        }
        return maxDiff;  
    } 
###### Rearrange an array with O(1) extra space
    void arrange(long long arr[], int n) {
        long long a[n];
        for(int i=0;i<n;i++)
        {   a[i]=arr[arr[i]];}
        for(int i=0;i<n;i++)
            arr[i]=a[i];
    }
###### Rearrange Array Alternately
    void rearrange(int *vec, int N) 
    { 
         int  maxelement= vec[N-1] +1;
               int maxIndex= N-1;
               int minIndex=0;
               for(int i=0; i < N ;i++){
                  if (i %2 == 0 ){
                      vec[i]+= (vec[maxIndex]%maxelement  )* maxelement;
                      maxIndex--;
                  }
                  else{
                       vec[i]+= (vec[minIndex]%maxelement  )* maxelement;
                       minIndex++;
                  }
               }
               for(int i=0;i<N;i++){
                   vec[i]= vec[i]/maxelement;
               }
    }
###### Smallest Positive missing number
    int missingNumber(int arr[], int n) {
        sort(arr,arr+n);
        int i;
        for(i=0;i<n;i++)
        {    if(arr[i]>0)
                break;
        }
        int p=0;
        for(int j=i;j<n;j++)
        {   if(arr[j]==p+1||arr[j]==arr[j-1])
                p=arr[j];
            else
                return p+1;
        }
        return p+1;
    }
###### Leaders in an array
    vector<int> leaders(int arr[], int n){
         vector<int> v;
         stack<int> s;
         int max_from_right=arr[n-1];
         s.push(max_from_right);

         for(int i=n-2;i>=0;i--)
         {
             if(arr[i]>=max_from_right)
             {
                 s.push(arr[i]);
                 max_from_right=arr[i];
             }
         }

         while(!s.empty())
         {
             v.push_back(s.top());
             s.pop();
         }    
        return v;
    }
###### Equilibrium Point
    int equilibriumPoint(long long a[], int n) {
                    int sum=0,leftsum=0,p=-1;
                    if(n==1)
                    return 1;
                    for(int i=0;i<n;i++)
                        sum+=a[i];
                    for(int i=0;i<n;i++)
                    {   sum=sum-a[i];
                        if(sum==leftsum)
                        {    p=i+1;break;}
                        leftsum=leftsum+a[i];
                    }

                    return p;
    }
###### Frequencies of Limited Range Array Elements
    void printfrequency(int arr[], int n)
    { 
        // Your code herre
        int *pCntr = new int[n];

        for(int i = 0; i < n; ++i)
        {
            pCntr[i] = 0;
        }

        for(int i = 0; i < n; ++i)
        {
            pCntr[arr[i]-1]++;
        }

        for(int i = 0; i < n; ++i)
        {
            cout << pCntr[i] << ' ';
        }

        delete pCntr;
    } 
###### Wave Array
    void convertToWave(int *arr, int n){
        if(n%2==1)
        {   for(int i=0;i<n-1;i=i+2)
                    swap(arr[i],arr[i+1]);
        }
        else
        {for(int i=0;i<n;i=i+2)
        {            swap(arr[i],arr[i+1]);}
        }
    }
###### Maximum occured integer
    int maxOccured(int L[], int R[], int n, int maxx){
        for(int i=0; i<=maxx; i++)
            arr[i] = 0;
        for(int i=0; i<n; i++)
        {
            arr[L[i]]++;
            arr[R[i]+1]--;
        }
        int ans = 0;
        int index;
        for(int i=1; i<=maxx; i++)
        {
            arr[i] += arr[i-1];
            if(ans < arr[i])
            {
                ans = arr[i];
                index = i;
            }
        }
        return index;
    }
###### Minimum adjacent difference in a circular array
    int minAdjDiff(int arr[], int n){    
        // Your code here
        int temp = 0;
        int minDiff = 0xFFFF;

        for(int i = 1; i < n; ++i)
        {
            temp = arr[i] - arr[i-1];
            if(minDiff > abs(temp)) minDiff = abs(temp);
        }

        temp = arr[n-1] - arr[0];
        if(minDiff > abs(temp)) minDiff = abs(temp);

        return minDiff;
    }
###### Reverse array in groups
    vector<long long> reverseInGroups(vector<long long> mv, int n, int k){
            vector<long long>v;
            stack<long long>stk;
            int i=0;int p=0;
            while(i<n)
            {   stk.push(mv[i]);
                p++;
                if(p==k)
                {   while(!stk.empty())
                    {   v.push_back(stk.top());
                        stk.pop();
                        p=0;
                    }
                }i++;
            }
            while(!stk.empty())
                    {   v.push_back(stk.top());
                        stk.pop();
                        //p=0;
                    }

    }
###### Strongest Neighbour
    void maximumAdjacent(int sizeOfArray, int arr[]){

        for(int i = 0; i < sizeOfArray - 1; ++i)
        {
            cout << (arr[i] < arr[i+1]? arr[i+1]:arr[i]) << ' ';
        }
    }
###### Max and Second Max
    vector<int> largestAndSecondLargest(int sizeOfArray, int arr[]){
        int max = INT_MIN, max2= INT_MIN;
         for(int i=0; i<sizeOfArray; i++){
             if(arr[i] > max)
                max = arr[i];
         }

         for(int i=0; i<sizeOfArray; i++){
             if(arr[i] == max){
                 arr[i] = -1;
             }
         }

         for(int i=0; i<sizeOfArray; i++){
             if(arr[i] > max2)
                max2 = arr[i];
         }
         vector <int> ans = {max, max2};
         return ans;
    }
## Searching
###### Median of Two sorted arrays
    int findMed(int a1[],int a2[],int n1,int n2){
        int begin=0,end=min(n1,n2);
        while(begin<=end){
            int i1=(begin+end)/2;
            int i2=(n1+n2+1)/2-i1;
            int min1= (i1==n1)?INT_MAX:a1[i1];
            int max1= (i1==0)?INT_MIN:a1[i1-1];
            int min2= (i2==n2)?INT_MAX:a2[i2];
            int max2= (i2==0)?INT_MIN:a2[i2-1];
            if(max1<=min2 && max2<=min1){
                if((n1+n2)%2==0){
                    return (max(max1,max2)+min(min1,min2))/2;
                }else{
                    return max(max1,max2);
                }
            }
            else if(max1>min2){
                end=i1-1;
            }
            else{
                begin=i1+1;
            }
        }
    }
###### Subarray with given sum
    void subarraySum(int arr[], int n, int s){

        // Your code here
        int i,sum=arr[0],start=0;
        for(i=1;i<=n;i++)
        {
            while(sum>s && start<i-1){
                sum=sum-arr[start];
                start++;  
            }
            if(sum==s){
            cout << start+1 << " " << i;
            return ;
            }
            if(i<n)
            sum=sum+arr[i];
        }
        cout << -1;
        return ;
    }
###### Allocate minimum number of pages
    bool possible(vector<int>&book,int m,int mid)
    {
        int stud=1;
        int count=0;
        int i=0;
        while(i<book.size())
        {
            if(count+book[i]>mid)
            {
                stud++;
                count=book[i];
                if(stud>m)
                {
                    return 0;
                }

            }
            else
            {
                count+=book[i];
            }
            i++;
        }
        return 1;
    }
    int ans(vector<int>&book,int m,int start,int end,int n)
    {
        int ans;
        int mid;
        while(start<=end)
        {
            mid=(start+end)/2;
            if(possible(book,m,mid))
            {
                ans=mid;
                end=mid-1;
            }
            else
            {
                start=mid+1;
            }
        }
        return start;

    }
###### Count More than n/k Occurences
    int countOccurence(int arr[], int n, int k)
    {
        sort(arr,arr+n);

        int i=0;
        int c = 0;
        int p = 0;
        while(i<(n-n/k))
        {
            if(arr[i+(n/k)] == arr[i])
            {

                if(p!=arr[i])
                {
                    c++;
                    p = arr[i];
                }

                i += (n/k) + 1;
            }
            else
                i++;
        }

        return c;
    }
###### Count only Repeated
        int n;
        cin>>n;

        int arr[n];
        int ans = 0;
        int num;
        for(int i=0; i<n; i++)
        {
            cin>>arr[i];
            if(arr[i]==arr[i-1] && i)
            {
                num = arr[i];
                ans++;
            }
        }

        cout<<num<<" "<<ans+1<<endl;
###### Smallest Positive missing number
    int missingNumber(int arr[], int n) { 

        // Your code here
        int H[1000000]={0};
        for(int i=0;i<n;i++) { if(arr[i]>0)
        H[arr[i]]++;
        }
        int x;
        for(int i=1;i<1000000;i++)
        {
        if(H[i]==0)
        {
        x=i;
        break;
        }
        }
        return x;

    }
###### Maximum Water Between Two Buildings
    int maxWater(int height[], int n) 
    { 
        int l = 0, r = n-1, m = 0;
        while (l < r) {
        int total = min(height[l], height[r]) * (r-l-1);
        m = max(m, total);

        if (height[l] <= height[r]) l++;
        else r--;
        }
        return m;


    }
###### Roof Top
    {
        int l = 0;
        int ans = 0;
        for(int i=1; i<n; i++)
        {
            if(a[i] <= a[i-1])
            {
                ans = max(ans,l);
                l=0;
            }
            else
                l++;
        }


        return max(ans,l);
    }
###### Two Repeated Elements
    void twoRepeated(int arr[], int n){
        for(int i = 0 ; i < n + 2 ; i++) {
            arr[((arr[i] % (n + 1)) - 1)] += (n + 1);
            if(arr[((arr[i] % (n+1)) - 1)] / (n+1) == 2) {
                    cout << ((arr[i] % (n+1)) - 1) + 1 << " ";       
            }
        }    
    }
###### Minimum Number in a sorted rotated array
    int minNumber(int arr[], int low, int high)
    {
        int n = high-low+1;
        while(low <= high)
        {
            int mid = (low+high)/2;

            if((mid==0 || arr[mid-1]>arr[mid]) && (mid==n-1 || arr[mid+1]>arr[mid]))
                return arr[mid];
            else if(arr[mid] > arr[high])
                low = mid+1;
            else 
                high = mid-1;
        }

    }
###### Floor in a Sorted Array
    int findFloor(vector<long long> v, long long n, long long x){

        int lo = 0;
        int hi = n-1;

        while(lo <= hi)
        {
            int mid = (lo+hi)/2;

            if(v[mid] <=x && (mid==n-1 || v[mid+1] > x))
                return mid;
            else if(v[mid] > x)
                hi = mid - 1;
            else
                lo = mid + 1;
        }

        return -1;
    }
###### Peak element
    int peakElement(int arr[], int n)
    {
       for(int i=0; i<n; i++){
           if(i==0 && arr[i]>arr[i+1]){
               return i;
           }
           else if(i==n-1 && arr[i]>arr[i-1]){
               return i;
           }
           else{
               if(arr[i-1]<arr[i] && arr[i]>arr[i+1])
               return i;
           }
       }
    }
###### Left Index
    int leftIndex(int sizeOfArray, int arr[], int elementToSearch){
        int low = 0;
        int high = sizeOfArray - 1;
        while(low <= high)
        {
            int mid = low + (high - low)/2;
            if((arr[mid] == elementToSearch) && ((mid == 0) || arr[mid-1] != elementToSearch))
            {
                return mid;
            }
            else if (arr[mid] >= elementToSearch)
            {
                high = mid - 1;
            }   
            else
            {
                low = low + 1;
            }
        }   
        return -1;
    }
## Sorting

###### Minimum Platforms
    int findPlatform(int arr[], int dep[], int n)
    {           if(n==1)
                    return 1;
                sort(arr,arr+n);
                sort(dep,dep+n);
                int ptfno=1;
                int i=1;int j=0;
                while(i<n && j<n)
                {   if(arr[i]>dep[j])
                    {    i++;j++;   }
                    else if(arr[i]<=dep[j])
                     {   ptfno++;i++;}
                }
                return ptfno;
        // Your code here
    }
###### Closer to sort
    int closer(int arr[],int n, int x)
    {
        int low=0,high=n-1;
        while(low<=high)
        {
            int mid=(low+high)/2;
            if(arr[mid]==x)
                return mid;
            if(arr[mid-1]==x)
                return mid-1;
            if(arr[mid+1]==x)
                return mid+1;
            else if(arr[mid]>x)
                high=mid-1;
            else
                low=mid+1;


        }
        return -1;
    }
###### Merge three sorted arrays
vector<int> mergeThree(vector<int>& A, vector<int>& B, vector<int>& C) 
{
    int m, n, o, i, j, k; 
    // Get Sizes of three vectors 
    m = A.size(); 
    n = B.size(); 
    o = C.size(); 

    // Vector for storing output 
    vector<int> D; 
    D.reserve(m + n + o); 

    i = j = k = 0; 

    while (i < m && j < n && k < o) { 

        // Get minimum of a, b, c 
        int minn = min(min(A[i], B[j]), C[k]); 

        // Put m in D 
        D.push_back(minn); 

        // Increment i, j, k accordingly
        if (minn == A[i]) 
            i++; 
        else if (minn == B[j]) 
            j++; 
        else
            k++; 
    } 

    // C has exhausted 
    while (i < m && j < n) { 
        if (A[i] <= B[j]) { 
            D.push_back(A[i]); 
            i++; 
        } 
        else { 
            D.push_back(B[j]); 
            j++; 
        } 
    } 

    // B has exhausted 
    while (i < m && k < o) { 
        if (A[i] <= C[k]) { 
            D.push_back(A[i]); 
            i++; 
        } 
        else { 
            D.push_back(C[k]); 
            k++; 
        } 
    } 

    // A has exhausted 
    while (j < n && k < o) { 
        if (B[j] <= C[k]) { 
            D.push_back(B[j]); 
            j++; 
        } 
        else { 
            D.push_back(C[k]); 
            k++; 
        } 
    } 

    // A and B have exhausted 
    while (k < o) 
        D.push_back(C[k++]); 

    // B and C have exhausted 
    while (i < m) 
        D.push_back(A[i++]); 

    // A and C have exhausted 
    while (j < n) 
        D.push_back(B[j++]); 

    return D;  
} 
###### Merge Without Extra Space
    void merge(int arr1[], int arr2[], int n, int m) 
    {           int i=n-1,j=0;
                while(i>=0&&j<m)
                {   if(arr1[i]>=arr2[j])
                    {    swap(arr1[i],arr2[j]);
                        i--;
                        j++;
                    }
                    else i--;
                }
                sort(arr1,arr1+n);
                sort(arr2,arr2+m);
    }
###### Number of pairs
    long long countPairs(int x[], int y[], int m, int n)
    {
       sort(x,x+m);
       int max1=x[m-1];
       sort(y,y+n);
       long long res=0;
       //for 1 and 2
       long long ct2_1=upper_bound(x,x+m,2)-lower_bound(x,x+m,2); //count of 2 in 1st array
       long long ct1_2=upper_bound(y,y+n,1)-lower_bound(y,y+n,1);  //count of 1 in 2nd array
       ct1_2+=y+n-lower_bound(y,y+n,5); 
       res+=ct2_1*ct1_2;
       long long ct3_1=upper_bound(x,x+m,3)-lower_bound(x,x+m,3);
       long long ct3_2=upper_bound(y,y+n,3)-lower_bound(y,y+n,3);
       res+=ct3_1*(n-ct3_2);
       ct1_2=upper_bound(y,y+n,1)-lower_bound(y,y+n,1);
       for(int i=4;i<=max1;i++)
       {
           long long ct1=upper_bound(x,x+m,i)-lower_bound(x,x+m,i);
           long long ct2=y+n-lower_bound(y,y+n,i+1);
           res+=ct1*(ct2+ct1_2);
       }
       return res;
    }
###### Kth smallest element
    int partition(int arr[], int l, int r) 
    { 
        int x = arr[r], i = l; 
        for (int j = l; j <= r - 1; j++) 
        { 
            if (arr[j] <= x) 
            { 
                swap(&arr[i], &arr[j]); 
                i++; 
            } 
        } 
        swap(&arr[i], &arr[r]); 
        return i; 
    } 

    int randomPartition(int arr[], int l, int r) 
    { 
        int n = r-l+1; 
        int pivot = rand() % n; 
        swap(&arr[l + pivot], &arr[r]); 
        return partition(arr, l, r); 
    }
    int kthSmallest(int arr[], int l, int r, int k) 
    { 
        // If k is smaller than number of elements in array 
        if (k > 0 && k <= r - l + 1) 
        {
            int pos = randomPartition(arr, l, r); 
            if (pos-l == k-1) 
                return arr[pos]; 
            if (pos-l > k-1)  // If position is more, recur for left subarray 
                return kthSmallest(arr, l, pos-1, k); 
            return kthSmallest(arr, pos+1, r, k-pos+l-1); 
        } 
        return INT_MAX; 
    }
###### Triplet Sum in Array
    bool find3Numbers(int arr[], int n, int x)
    {
        sort(arr,arr+n); 
        for(int i=0;i<n-2;i++)
        {
            int l=i+1; 
            int r=n-1; 
            int y=arr[i]; 
            while(l<r)
            {
                if(y+arr[l]+arr[r]==x)
                 return true; 
                else if(y+arr[l]+arr[r]<x)
                 l++; 
                else 
                 r--;
            }
        } 
        return false;
    }
###### Sort by Absolute Difference
    bool mycomp(pair<int,int> p1,pair<int,int> p2)
    {
        return (p1.first<p2.first);
    }

    void sortABS(int A[],int N, int k)
    {
       //Your code here
       pair<int,int> p[N];
       int diff;
       for(int i=0;i<N;i++)
       {
           diff=abs(A[i]-k);
           p[i].first=diff;
           p[i].second=A[i];
       }

       stable_sort(p,p+N,mycomp);

       for(int i=0;i<N;i++)
       {
           A[i]=p[i].second;
       }
    }
###### Closet 0s 1s and 2s
    void segragate012(int A[], int N)
    {
        int low = 0, high = N-1, mid = 0;

        while(mid <= high){

            // if any element appears to be 0
            // push that element to start
            if(A[mid]==0)
                swap(A[mid++], A[low++]);

            // if element found to be 1
            // push that to the mid
            else if(A[mid]==1)
                 mid++;

            // if element found to be 2,
            // push it to high
            else
                swap(A[mid], A[high--]);
            }

    }
###### Three way partitioning
    {
        vector<int> v[3];
        vector<int> ans;

        for(vector<int>::iterator it=A.begin();it!=A.end();it++){

            if(*it < lowVal) v[0].push_back(*it);
            else if((*it >= lowVal) &&(*it <= highVal )) v[1].push_back(*it);
              else v[2].push_back(*it);
        }


        for(int i=0;i<3;i++){

            for(vector<int>::iterator it=v[i].begin();it!=v[i].end();it++)
              {
                  ans.push_back(*it);
                  //cout<<ans.back()<<" ";
              }
        }

        return ans;
    }
###### Find triplets with zero sum
    bool findTriplets(int arr[], int n)
    {      int i,j,p,q=0,k;
            sort(arr,arr+n);
            if(n<3)
            {   q++; return false;}
            else
            {   for(k=0;k<n-3;k++)
                {   p=0-arr[k];
                    i=k+1,j=n-1;
                    while(i<j)
                    {   if(p==(arr[i]+arr[j]))
                        {  q++;  return true;}
                        else if(p>(arr[i]+arr[j]))
                            i++;
                        else
                            j--;
                    }
                }
            }
            if(q==0)
                return false;
        //Your code here
    }
###### Intersection of two sorted arrays
    void printIntersection(int arr1[], int arr2[], int N, int M) 
    { 
         int i=0,j=0;
         bool isCommonPresent=false;  
         while( i < N && j < M){
             if ( i> 0  && arr1[i] == arr1[i-1]) {  
                 i++;
                 continue;
             }      
             if ( arr1[i] < arr2[j])
                    i++;
             else if ( arr1[i] > arr2[j])
                    j++;
             else{
                 cout<<arr1[i]<<" ";
                 i++;
                 j++;
                 isCommonPresent=true;
             }        
         }
         if( !isCommonPresent)
             cout<<"-1";     
    }
###### Union of Two Sorted Arrays
    vector<int> findUnion(int arr1[], int arr2[], int n, int m)
    {
        vector<int> v;
        int i=0,j=0;
        while(i<n && j<m){
            if(arr1[i]>arr2[j]){
                v.push_back(arr2[j]);
                j++;
            }
            else if(arr1[i]<arr2[j]){
                v.push_back(arr1[i]);
                i++;
            }
            else{
                v.push_back(arr1[i]);
                v.push_back(arr2[j]);
                i++;
                j++;
            }
        }
        while(i<n) v.push_back(arr1[i++]);
        while(j<m) v.push_back(arr2[j++]);

        vector<int> v1;
        v1.push_back(v[0]);
        for(int k=1;k<v.size();k++){
            if(v[k-1]!=v[k]){
                v1.push_back(v[k]);
            }
        }
        return v1;

    }
###### Inversion of array
    long long my_counter = 0;
    void merge(long long a[], long long p, long long q, long long r){
        long long l = q-p+1;
        long long a1[l];
        long long l2 = r-q;
        long long a2[l2];
        for(long long i = 0;i<l;i++){
            a1[i] = a[i+p];
        }
        for(long long i = 0;i<l2;i++){
            a2[i] = a[q+i+1];
        }
        long long left = 0, right = 0, k = p;
        while(left < l && right < l2)
        {
            if(a1[left] <= a2[right]){
                a[k] = a1[left];
                left++;
            }
            else{
                a[k] = a2[right];
                right++;
                my_counter += (l-left); // Increementing counter
            }
            k++;
        }
        while(left < l){
            a[k++] = a1[left++];
        }
        while(right < l2){
            a[k++] = a2[right++];
        }
    }
    void mergeSort(long long a[], long long p, long long r)
    {
        if(p < r)
        {
            long long q = (p+r)/2;
            mergeSort(a, p, q);
            mergeSort(a, q+1, r);
            merge(a, p, q, r);
        }
    }
    long long int inversionCount(long long A[],long long N)
    {
        mergeSort(A,0,N-1);
        long long int res = my_counter;
        my_counter = 0;
        return res;
    }
## Matrix
###### Adding two matrices
    void sumMatrix(int n1, int m1, int n2, int m2, int arr1[SIZE][SIZE], int arr2[SIZE][SIZE]){
        if(n1!=n2 || m1!=m2){
            cout<<-1;
        }
        else{
        for(int i=0;i<n1;i++){
            for(int j=0;j<m1;j++){
                cout<<arr1[i][j]+arr2[i][j]<<' ';
            }
        }
        }
    }
###### Sum of upper and lower triangles
    void sumTriangles(int n, int mat[SIZE][SIZE]){
        int u=0,d=0;
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++){
                u=u+mat[i][j];
            }
        }
        for(int i=n-1;i>=0;i--){
            for(int j=i;j>=0;j--){
                d=d+mat[i][j];
            }
        }
        cout<<u<<' '<<d;
    }
###### Multiply the matrices
    void multiplyMatrix(int n1, int m1, int n2, int m2, long long arr1[SIZE][SIZE], long long arr2[SIZE][SIZE]){
        if(m1!=n2)
        {
            cout<<"-1";
        }
        else
        {
            int sum=0;
           for(int i=0;i<n1;i++)
           {
               for(int j=0;j<m2;j++)
               {
                   sum=0;
                   for(int k=0;k<n2;k++)
                   {
                       sum+=arr1[i][k]*arr2[k][j];
                   }
                   cout<<sum<<" ";
               }
           }
        }
    }
###### Print Matrix in snake Pattern
void print(int mat[][100],int n)
{
   for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            for (int j = 0; j < n; j++)
                cout << mat[i][j] << " ";
        } else {
            for (int j = n - 1; j >= 0; j--)
                cout << mat[i][j] << " ";
        }
    }
}
###### Transpose of Matrix
    void transpose( int A[][N],int n) 
    {       int p;
            for(int i=0;i<n;i++)
            {   for(int j=i;j<n;j++)
                {       if(i!=j)
                        {   p=A[j][i];
                            A[j][i]=A[i][j];
                            A[i][j]=p;
                        }
                        else;
                }
            }
    }
###### Rotate by 90 degree
    void rotateby90(int n, int a[][N]){
        for(int i=0;i<n;i++){
            reverse(a[i],a[i]+n);
            }
        for(int i=0;i<n;i++){ 
            for(int j=0;j<n;j++)
                {if(j>i)
                   swap(a[i][j],a[j][i]);
                }
        } 
    }
###### Determinant of a Matrix
    int determinantOfMatrix( int mat[N][N], int n)
    {

        if(n==1)
        return mat[0][0];
        int i,j,arr[N][N],sum=0,k;
        for(j=0;j<n;j++)
        {
            for(i=1;i<n;i++)
            for(k=0;k<n;k++)
            {
                if(j==k)
                k++;
                if(j<k)
                arr[i-1][k-1]=mat[i][k];
                else
                arr[i-1][k]=mat[i][k];
            }
            sum=sum+pow(-1,j)*(determinantOfMatrix(arr,n-1))*mat[0][j];
        }
        return sum;
    }
###### Boundary traversal of matrix
    void boundaryTraversal(int n1, int m1, int arr[SIZE][SIZE]){
       for(int i=0;i<m1;i++)
           cout<<arr[0][i]<<" ";
        for(int i=1;i<n1;i++)
          cout<<arr[i][m1-1]<<" ";
        if(n1>1)
         for(int i=m1-2;i>=0;i--)
           cout<<arr[n1-1][i]<<" ";   
        if(m1>1)
         for(int i=n1-2;i>0;i--)
          cout<<arr[i][0]<<" ";   
    }
###### Exchange matrix columns
    void exchangeColumns(int n1, int m1, int arr1[SIZE][SIZE]){
        for(int i=0;i<n1;i++){
            swap(arr1[i][0],arr1[i][m1-1]);
        }

        for(int i=0;i<n1;i++){
            for(int j=0;j<m1;j++){
                cout<<arr1[i][j]<<' ';
            }
        }   
    }
######  Spirally traversing a matrix
    void spirallyTraverse(int m, int n, int a[SIZE][SIZE]){
        int r1=0;
        int r2=m-1;
        int c1=0;
        int c2=n-1;
        while(r1<=r2&&c1<=c2)
        {  
            for(int i=c1;i<=c2;i++)
            cout<<a[r1][i]<<" ";

            for(int j=r1+1;j<=r2;j++)
            cout<<a[j][c2]<<" ";

            if(r1!=r2)
            for(int i=c2-1;i>=c1;i--)
            cout<<a[r2][i]<<" ";

            if(c1!=c2)
            for(int j=r2-1;j>r1;j--)
            cout<<a[j][c1]<<" ";

            r1++;
            c1++;
            r2--;
            c2--;
        }
    }
###### Reversing the columns of a Matrix
    void reverseCol(int n1, int m1, int arr1[SIZE][SIZE])
    {
        int l=0;
        int h=m1-1;
        while(l<h){
            for(int i=0;i<n1;i++){
                swap(arr1[i][l],arr1[i][h]);
            }
            l++;
            h--;
        }
    }
###### Interchanging the rows of a Matrix
    void interchangeRows(int n1, int m1, int arr1[SIZE][SIZE])
    {
        int low=0,high=n1-1;
        while(low<=high)
        {
            for(int i=0;i<m1;i++)
            {
                int temp=arr1[low][i];
                arr1[low][i]=arr1[high][i];
                arr1[high][i]=temp;
            }
            low++;
            high--;
        }
    }
###### Search in a row-column sorted Matrix
    int search( int n,int m, int x, int mat[SIZE][SIZE])
    {
       int i=0,j=m-1;
       while(i<n&&j>=0)
       {
           if(mat[i][j]==x)
           return 1;
           else if(x<mat[i][j])
           j--;
           else
           i++;
       }
       return 0;
    }
###### Boolean Matrix
    void booleanMatrix(int r, int c, int a[SIZE][SIZE])
    {
        int i,j,A[r],B[c];
        memset(A,0,sizeof(A));
        memset(B,0,sizeof(B));
        for(i=0;i<r;i++)
        {
            for(j=0;j<c;j++)
                if(a[i][j])
                {
                    A[i]=1;
                    break;
                }
        }
        for(i=0;i<c;i++)
        {
            for(j=0;j<r;j++)
                if(a[j][i])
                {
                    B[i]=1;
                    break;
                }
        }
        for(i=0;i<r;i++)
        {
            for(j=0;j<c;j++)
                cout<<max(A[i],B[j])<<" ";
            cout<<endl;
        }
    }
###### Make Matrix Beautiful
    int findMinOpeartion(int matrix[][100], int n)
    {

        int sumRow[n], sumCol[n]; 
        memset(sumRow, 0, sizeof(sumRow)); 
        memset(sumCol, 0, sizeof(sumCol));
        for (int i = 0; i < n; ++i) 
            for (int j = 0; j < n; ++j) { 
                sumRow[i] += matrix[i][j]; 
                sumCol[j] += matrix[i][j]; 
            } 
        int maxSum = 0; 
        for (int i = 0; i < n; ++i) { 
            maxSum = max(maxSum, sumRow[i]); 
            maxSum = max(maxSum, sumCol[i]); 
        } 
        int count = 0; 
        for (int i = 0, j = 0; i < n && j < n;) { 
            int diff = min(maxSum - sumRow[i], 
                           maxSum - sumCol[j]); 
            matrix[i][j] += diff; 
            sumRow[i] += diff; 
            sumCol[j] += diff; 
            count += diff; 
            if (sumRow[i] == maxSum) 
                ++i; 
            if (sumCol[j] == maxSum) 
                ++j; 
        } 
        return count;
    }
## Hashing
###### Separate chaining in Hashing
    {
       for(int i=0; i<sizeOfArray; i++)
            hashTable[arr[i]%hashSize].push_back(arr[i]);   
    }
######  Linear Probing in Hashing 
    void linearProbing(int hash[],int hashSize,int arr[],int sizeOfArray)
    {
        //Your code here
        for(int i=0; i<sizeOfArray; i++){
            int hashVal = arr[i]%hashSize;
            int temp = hash[hashVal];
            while(hash[hashVal]!=-1){
                hashVal++;
                if(temp==hash[hashVal])
                    return;
                if(hashVal==hashSize)
                    hashVal=0;
            }
            hash[hashVal]=arr[i];
        }
    }
###### Quadratic Probing in Hashing
    {
        for(int i=0; i<sizeOfArray; i++)
        {
            int k = 0;
            while(hash[(arr[i]+k*k)%hashSize] != -1)
                k++;
            hash[(arr[i]+k*k)%hashSize] = arr[i];
        }
    }
###### First Repeating Element
    int firstRepeated(int arr[], int n) {
               set<int>set;
               int min=-1;
               for(int i=n-1;i>=0;i--)
               {    if(set.find(arr[i])!=set.end())
                            min=i+1;
                    else
                        set.insert(arr[i]);
               }
               return min;
    }
###### Intersection of two arrays
    int NumberofElementsInIntersection (int a[], int b[], int n, int m )
    {    int count=0;
            unordered_set<int>s;
            for(int i=0;i<n;i++)
                s.insert(a[i]);
            for(int i=0;i<m;i++)
            {   if(s.find(b[i])!=s.end())
                {   s.erase(b[i]);
                        count++;
                }
            }
         return (count);
    }
###### Union of two arrays
    int doUnion(int a[], int n, int b[], int m)  {
        unordered_set<int> s;
        for(int i=0;i<n;i++){
            s.insert(a[i]);
        }
        for(int i=0;i<m;i++){
            s.insert(b[i]);
        }
        return s.size();
    }
###### Subarray with 0 sum
    bool subArrayExists(int arr[], int n)
    {
        unordered_set<int> prefixSum;
            int sum = 0;
            for ( int i=0;i < n ; i++){
                sum+=arr[i];
                if ( prefixSum.find(sum) != prefixSum.end() ||  0  == sum)
                   return true;
                prefixSum.insert(sum);
            }
            return false;
    }
###### Check if two arrays are equal or not
        map<int , int> m;
	    for(i=0;i<n;i++){
	        cin >> a[i] ; 
	        m[a[i]]++ ;
	    }
	    for(i=0;i<n;i++){
	        cin >> b[i] ;
	        m[b[i]]-- ;
	    }
	    int f = 0 ;
	    for(auto it=m.begin();it!=m.end();it++){
	        if(it->second > 0){
	            f = 1 ;
	            break ;
	        }
	    }
	    
	    cout << !f << "\n" ;
###### Subarray range with given sum
    {
        unordered_map<int,int> m;
        m[0] = 1;

        int c = 0;
        int s = 0;
        for(int i=0; i<n; i++)
        {
            s += arr[i];
            if(m.find(s-sum) != m.end())
                c += m[s-sum];

            m[s]++;
        }

        return c;
    }

######  Subarrays with equal 1s and 0s
    long long int countSubarrWithEqualZeroAndOne(int arr[], int n)
    {
        //Your code here
        for(int i=0;i<n;i++)
        { 
            if(arr[i]==0)
            {arr[i]=-1; } }
            long long int c=0; int sum=0;
            unordered_map<int,int>m;
        for(int i=0;i<n;i++)
        { sum=sum+arr[i]; if(sum==0){c++;}
        if(m.count(sum)>=0)
        {
        c+=m[sum];
        }
        m[sum]++;
        }
        return c;
    }
###### Positive Negative Pair
        map<int,int> p;
	    map<int,int> n;
	    
	    int x;
	    for(int i=0; i<num; i++)
	    {
	        cin>>x;
	        
	        if(x>0)
	            p[x]++;
	        else
	            n[abs(x)]++;
	    }
	    
	    bool flag = false;
	    for(auto i:p)
	    {
	        int r = min(n[i.first],i.second);
	        for(int j=0; j<r; j++)
	        {
	            flag = true;
	            cout<<i.first<<" "<<-1*i.first<<" ";
	        }
	    }
	    
	    if(!flag)
	        cout<<0;
	    
	    cout<<endl;
###### Sorting Elements of an Array by Frequency
    bool sortbysec(pair<int,int> a, pair<int,int> b)
    {
        if(a.second!=b.second) return a.second>b.second;
        else return a.first<b.first;
    }
    void sortByFreq(int arr[],int n)
    {
        //Your code here
        unordered_map<int,int> m;
        for(int i=0;i<n;i++) m[arr[i]]++;
        vector<pair<int,int>> v;
        for(auto it=m.begin();it!=m.end();it++)
            v.push_back(make_pair(it->first,it->second));
        sort(v.begin(),v.end(),sortbysec);
        for(auto it=v.begin();it!=v.end();it++)
        {
            int k=it->second;
            while(k--) cout<<it->first<<" ";
        }
    }
###### Zero Sum Subarrays
     int sum=0;
     map<int,int> m;
     int cnt=0;
     for(int i=0;i<n;i++)
     {
         sum=sum+a[i];
         if(m.find(sum)==m.end())
         {
             m[sum]++;
             if(sum==0)
             cnt++;
         }
         else
         {
             cnt+=m[sum];
             m[sum]++;
             if(sum==0)
             cnt++;
         }
     }
     cout<<cnt<<"\n";
###### Longest consecutive subsequence
    {   map<int,int> m;
        for(int i=0; i<n; i++)
            m[arr[i]]++;
        bool flag = 0;
        int tmp;
        int ans = 0;
        int curr = 0;
        for(auto i:m)
        {
            if(!flag || i.first == tmp+1)
            {
                curr++;
                flag = 1;
                tmp = i.first;
            }
            else
            {
                ans = max(ans,curr);
                curr = 1;
                tmp = i.first;
            }
        }
        ans = max(ans,curr);
        return ans;
    }
###### Relative Sorting
        for(int i=0;i<n1;i++)
	        h[A[i]]++;
	    for(int i = 0;i<n2;i++)
	    {
	        if(h.find(B[i])!=h.end())
	        {
	            while(h[B[i]]>0)
	            {
	                cout << B[i] << " ";
	                h[B[i]]--;
	            }
	            h.erase(B[i]);
	        }
	    }
	    for(it = h.begin();it!=h.end();it++)
	    {
	        while((it->second)-- != 0)
	            cout << it->first << " ";
	    }
	    cout << endl;
## Strings
###### Naive Pattern Search
	bool search(string pat, string txt) 
	{ 
		int t=txt.find(pat);
	if(t>=0)
	return true;
	return false;

	}
###### Distinct Pattern Search
	bool search(string pat, string txt) 
	{ 
	    if(txt.find(pat) != string::npos)
		return true;

	    return false;

	}
###### Binary String
	long binarySubstring(int n, string a){

	    auto ones = std::count(a.begin(), a.end(), '1');
	    return ones * (ones - 1) / 2;
	}
###### Implement strstr
     for(i=0;i<m;i++)
     {
         s2=s.substr(i,n);
         if(s2.compare(x)==0)
         {
             return i;
         }
     }
###### Check if string is rotated by two places
	bool isRotated(string str1, string str2)
	{
	    int l = str1.length();

	    bool flag1 = true;
	    bool flag2 = true;
	    for(int i=0; i<l; i++)
	    {
		if(str1[(i+(l-2))%l] != str2[i])
		    flag1 = false;

		if(str1[(i+2)%l] != str2[i])
		    flag2 = false;
	    }

	    return flag1||flag2 && (str1.length() == str2.length());
	}
###### Check if strings are rotations of each other or not
	bool areRotations(string s1,string s2)
	{                   
	    string s3=s1+s1;
	    /*int n=s1.size()-1,p=0;
	    for(int i=0;i<s1.size();i++)
	    {   //s4=s3.substr(i,i+n);
		if(s2==s3.substr(i, i+n))
		{    p=1;break;}
	    }
	    if(p==1)
		return 1;
	    else return 0;*/
	    if(s1.size()!=s2.size())
		    return 0;
	    return (s3.find(s2)!=string::npos);
	}
###### Isomorphic Strings
	bool areIsomorphic(string str1, string str2)
	{
	    if(str1.size()!=str2.size()){
		return 0;
	    }
	    int temp1[256]={0};
	    int temp2[256]={0};

	    for(int i=0;i<str1.length();i++)
	    {
		temp1[str1[i]]++;
		temp2[str2[i]]++;

		if(temp1[str1[i]]==temp2[str2[i]])
		    continue;
		else
		    return false;
	    }
	    return true;
	}
###### Check if a string is Isogram or not
	{
	    std::unordered_set<char> m;
	    for (auto c : s)
	    {
		auto it = m.find(c);
		if (it != m.end())
		    return 0;
		m.insert(c);
	    }
	    return 1;
	}
###### Keypad typing
     l=strlen(str) ;
     for(i=0;i<l;i++)
     {
         if(str[i]>='a'&&str[i]<='c')
             cout<<2 ;
        else if(str[i]>='d'&&str[i]<='f')
             cout<<3 ;
        else if(str[i]>='g'&&str[i]<='i')
             cout<<4 ;
        else if(str[i]>='j'&&str[i]<='l')
             cout<<5 ;
        else if(str[i]>='m'&&str[i]<='o')
             cout<<6 ;
        else if(str[i]>='p'&&str[i]<='s')
             cout<<7 ;
        else if(str[i]>='t'&&str[i]<='v')
             cout<<8 ;
        else if(str[i]>='w'&&str[i]<='z')
             cout<<9 ;
     }
     cout<<endl ;
###### Repeating Character - First Appearance Leftmost
	int repeatedCharacter (string s) 
	{ 
	    //Your code here
	    for(int i = 0; i < s.length(); i++)
	    {
		if(s.find(s[i], i+1) != string::npos)
		    return i;
	    }
	return -1;
	} 
###### Non Repeating Character
	char nonrepeatingCharacter(string S)
	{
	    unordered_map<char,int> mp;
	    for(char c : S)
		mp[c]++;
	    for(char c :S)
		if(mp[c] == 1)
		    return c;
	    return '$';

	}
###### Maximum Occuring Character
	char getMaxOccuringChar(char* str)
	{                   
	    map<char,int>map;
	    for(int i=0;str[i]!='\0';i++)
		   map[str[i]]++;
	    int count=0;char ch;
	    for( auto i:map )
	    {   if(i.second>count)
		{   count=i.second;
		    ch=i.first;
		}
	    }
	    return ch;
	}
###### Remove common characters and concatenate
	string concatenatedString(string s1, string s2) 
	{ 
		string ans = "";
		for(int i=0; i < s1.length(); i++)
		if(s2.find(s1[i])==string::npos)
		ans += s1[i];
		for(int i=0; i < s2.length(); i++)
		if(s1.find(s2[i])==string::npos)
		ans += s2[i];
		return ans;
	}
###### Reverse words in a given string
	void reverseWords(char *s) {    
	   int count=0,i=0;stack<char>stk;
	   while(s[i]!='\0'){count++;i++;}
		for(int i=count-1;i>=0;i--)
		{   if(s[i]!='.')
				stk.push(s[i]);
			else
				{   while(!stk.empty())
					{   cout<<stk.top();
						stk.pop();
					}
					cout<<".";
				}
		}
		while(!stk.empty())
					{   cout<<stk.top();
						stk.pop();}
	}
###### Sum of numbers in string
	int findSum(string ch)
	{
		string temp = "";
		int sum = 0;
		for (int i=0;ch[i]!='\0';i++)
		{
			if (isdigit(ch[i]))
				temp += ch[i];
			else
			{
				sum += atoi(temp.c_str());
				temp = "";
			}
		}
		return sum+atoi(temp.c_str()) ;
	}
###### Minimum indexed character
	void printMinIndexChar(string str, string patt)
	{       int s1=str.size(),s2=patt.size(),min=max;
			for(int i=0;i<s2;i++)
			{   for(int j=0;j<s1;j++)
				{   if(str[j]==patt[i])
					{   if(j<min)
							min=j;
					}
				}
			}
			if(min!=max)
				cout<<str[min];
			else cout<<"No charecter present";
	}
###### Smallest window in a string containing all the characters of another string
	string smallestWindow (string S, string P)
	{
		if (S.length() < P.length()) 
		{ 
			return "-1"; 
		} 
		int hash_pat[256] = {0}; 
		int hash_str[256] = {0};
		for (int i = 0; i < P.length(); i++) 
			hash_pat[P[i]]++; 
		int start = 0, start_index = -1, min_len = INT_MAX;
		int count = 0; 
		for (int j = 0; j < S.length() ; j++) 
		{ 
			hash_str[S[j]]++; 
					if (hash_pat[S[j]] != 0 && 
				hash_str[S[j]] <= hash_pat[S[j]] ) 
				count++; 
			if (count == P.length()) 
			{ 

				while ( hash_str[S[start]] > hash_pat[S[start]] 
					|| hash_pat[S[start]] == 0) 
				{ 

					if (hash_str[S[start]] > hash_pat[S[start]]) 
						hash_str[S[start]]--; 
					start++; 
				} 
				int len_window = j - start + 1; 
				if (min_len > len_window) 
				{ 
					min_len = len_window; 
					start_index = start; 
				} 
			} 
		} 
		if (start_index == -1) 
		{ 
		return "-1"; 
		} 

		return S.substr(start_index, min_len); 
	}
###### Nth number made of prime digits
	string nthprimedigitsnumber(int number) 
	{ 
		int rem; 
		string num; 
		while (number) { 
			// remainder for check element position 
			rem = number % 4; 
			switch (rem) { 
			case 1: 
				num.push_back('2'); 
				break; 
			case 2: 
				num.push_back('3'); 
				break;  
			case 3: 
				num.push_back('5'); 
				break;  
			case 0: 
				num.push_back('7'); 
				break; 
			} 
			number--;
			number = number / 4; 
		} 
		  reverse(num.begin(), num.end());
		return num; 
	}
###### The Modified String
	int modified (string a){
		int n=0, insert=0;
		for(int i=1; i<a.length(); i++){
			if(a[i]==a[i-1])
				n++;
			else{
				insert+=n/2;
				n=0;
			}
		}
		insert+=n/2;
		return insert;
	}
###### Case-specific Sorting of Strings
	string caseSort(string str, int n){
		string str1 , str2= str;
		sort(str2.begin(),str2.end());
		int lowIndex= count_if(str2.begin(),str2.end(),::isupper);
		for(int i=0,j1=0,j2=lowIndex;i<str.length();i++)
		{
			if(islower(str[i]))
			{
				str1+=str2[j2++];
			}
			else
			{
				str1+=str2[j1++];
			}
		}
		return str1;
	}
###### Lexicographic Rank Of A String
	int findRank(string S) 
	{
		int res=1;
		int count[256]={0};
		int mul=fact(S.length());
		for(int i=0;i<S.length();i++){
			if(count[S[i]]==1)
			   return 0;
			count[S[i]]++;
		}
		for(int i=1;i<256;i++){
			count[i]=count[i]+count[i-1];
		}
		for(int i=0;i<S.length();i++){
			mul=mul/(S.length()-i);
			res=res+count[S[i]-1]*mul;
			for(int j=S[i];j<256;j++){
				count[j]--;
			}
		}
		return res;
	}
###### Rabin Karp - Pattern Searching
	bool search(string pat, string txt, int q) 
	{ 
		int n=txt.length();
		int m=pat.length();
		int h=1;
		for(int i=1;i<m;i++)
		{
			h=(h*d)%q;
		}
		int p=0,t=0;
		for(int i=0;i<m;i++)
		{
			p=(p*d+pat[i])%q;
			t=(t*d+txt[i])%q;
		}
		for(int i=0;i<=n-m;i++)
		{
			if(p==t)
			{
				bool flag=true;
				for(int j=0;j<m;j++)
					if(txt[i+j]!=pat[j])
					{
						flag=false;
						break;
					}
				if(flag==true)
					return true;
			}
			if(i<n-m)
			{
				t=((d*(t-txt[i]*h)+txt[i+m]))%q;
				if(t<0)
					t=t+q;
			}
		}
		return false;
	}
###### Pattern Search KMP
	void computeLPSArray(string pat, int M, int* lps) 
	{ 
		int len=0;
		lps[0]=0;
		int i=1;
		while(i<M)
		{
			if(pat[i]==pat[len])
			{
				len++;
				lps[i]=len;
				i++;
			}
			else
			{
				if(len==0)
				{
					lps[i]=0;
					i++;
				}
				else
				{
					len=lps[len-1];
				}
			}
		}
	} 
	bool KMPSearch(string pat, string txt) 
	{
		int N=txt.length();
		int M=pat.length();
		int lps[M];
		computeLPSArray(pat,pat.length(),lps);
		int i=0,j=0;
		while(i<N)
		{
			if(pat[j]==txt[i])
			{
				i++;
				j++;
			}
			if(j==M)
			{
				return true;
				j=lps[j-1];
			}
			else if(i<N && pat[j]!=txt[i])
			{
				if(j==0)
				{
					i++;
				}
				else
				{
					j=lps[j-1];
				}
			}
		}
		return false;    
	}
###### Delete without head pointer
	void deleteNode(Node *node)
	{
	   while(node->next->next != NULL){
		   node->data = node->next->data;
		   node = node->next;     
	   }
		node->data = node->next->data;
		node->next = NULL;
	}
###### Remove duplicates from an unsorted linked list
	Node *removeDuplicates(Node *root)
	{
		Node *p=root,*q=NULL;
		unordered_set<int>hash;

		while(p!=NULL){
			if(hash.find(p->data)==hash.end()){
				hash.insert(p->data);
				q=p;
			}
			else{
				q->next=p->next;
				delete p;
			}
			p=q->next;
		}
		return root;

	}
###### Merge two sorted linked lists
	Node* sortedMerge(Node* head_A, Node* head_B)  
	{           Node *head;
				Node **ref=&head;
				while(head_A && head_B)
				{   if(head_A->data < head_B->data)
					{   *ref=head_A;
						head_A=head_A->next;
					}
					else
					{   *ref=head_B;
						head_B=head_B->next;
					}
					ref=&((*ref))->next;
				}
				*ref=(head_A)?head_A:head_B;
				return head;
	}
###### Swap Kth nodes from ends
	Node *swapkthnode(Node* head, int n, int k)
	{
		// Your Code here
		if(k>n) return head;
		if(2*k-1 == n) return head;
		Node *x = head;
		Node *x_prev= NULL;
		for(int i=1;i<k;i++)
		{
			x_prev = x;
			x = x->next;
		}
		Node *y = head;
		Node *y_prev= NULL;
		for(int i=1;i<n-k+1;i++)
		{
			y_prev = y;
			y = y->next;
		}
		if(x_prev) x_prev->next=y;
		if(y_prev) y_prev->next=x;
		Node *temp = x->next;
		x->next = y->next;
		y->next = temp;
		if(k==1) head= y;
		if(k==n) head= x;
		return head;
	}
###### Detect Loop in linked list
	int detectloop(Node *head) {
		Node *x,*x2;
		x=head;
		x2=head;
		int c=0;
		while(x2!=NULL && x2->next!=NULL){
			x=x->next;
			x2=x2->next->next;
			if(x==x2){
			c=1;
			break;
			}
		}
		return c;
   }
###### Find length of Loop
	int countNodesinLoop(struct Node *head)
	{
		 int count=0;
		 if(head==NULL) return 0;
		 Node *fast=head, *slow=head;
		 while(fast!=NULL && fast->next!=NULL)
		 {
			  fast=fast->next->next;
			 slow=slow->next;
			 if(fast==slow)
			 {
				 Node *temp=slow;
				 while(temp->next!=fast)
				 {
					 count++;
					 temp=temp->next;
				 }
				 return count+1;
			 }     
		 }
		 return count;
	}
###### Remove loop in Linked List
	void removeTheLoop(Node *head)
	{
		Node *slow=head;
		Node *fast=head->next;
		Node *prev=NULL;
		Node *ptr2=NULL;
		while(slow && fast && fast->next){
			prev=slow;
			slow=slow->next;
			fast=fast->next->next;
			if(slow==fast){

				Node *ptr=head;
				while(1){
					ptr2= slow;
					 while(ptr2->next!=slow && ptr2->next!=ptr)
						ptr2=ptr2->next;

					if(ptr2->next==ptr)
					break;

					ptr=ptr->next; 
				}
				ptr2->next=NULL;
				break;
			}

		}
	}
###### Rotate a Linked List
	void rotate(struct node **head_ref, int k)
	{ 
		 struct node * end = *head_ref;
		 while(end->next != NULL) end = end->next;
		 while(k--){
			 end->next = *head_ref;
			 end = end->next;
			 (*head_ref) = (*head_ref)->next;
		 }
		 end->next = NULL;
	}
###### Add two numbers represented by linked lists
	Node* addTwoLists(Node* first, Node* second) 
	{
		Node *res =NULL, *prev = NULL, *temp;
		int carry = 0, sum;
		while(first !=NULL || second != NULL)
		{
			sum = carry + (first? first->data: 0) + (second? second->data: 0);
			carry = (sum>=10 ? 1:0);
			sum = sum%10;
			temp = new Node(sum);
			if(res == NULL)
				res =temp;
			else
				prev->next =temp;
			prev = temp;
			if(first) 
				first = first->next;
			if(second)
				second = second->next;
		}
		if(carry>0)
			temp->next = new Node(carry);
		return res;
	}
###### Pairwise swap of nodes in LinkeList
	struct Node* pairwise_swap(struct Node* head)
	{
	   Node *newhead; 
	   Node *nex1; 
	   if(head==NULL||head->next==NULL)
	   return head; 
	   nex1=head->next->next; 
	   newhead=head->next;
	   newhead->next=head; 
	   head->next=pairwise_swap(nex1); 
	   return newhead;
	}
###### Check if Linked List is Palindrome
	bool isPalindrome(Node *head)
	{
		Node *curr = head;
		stack<int> st;
		while(curr!=NULL)
		{
			st.push(curr->data);
			curr = curr->next;
		}
		curr = head;
		while(curr!=NULL)
		{
			if(st.top() != curr->data)
				return 0;
			st.pop();
			curr = curr->next;
		}
		return 1;
	}
###### Merge Sort for Linked List
	Node* midPoint(Node* a) {
		if(a == NULL || a->next == NULL) {
			return a;
		}
		Node* slow = a;
		Node* fast = a->next;
		while(fast != NULL && fast->next != NULL) {
			fast = fast->next->next;
			slow = slow->next;
		}
		return slow;
	}
	Node* merge(Node* a, Node* b) {
		if(a == NULL) {
			return b;
		}
		else if(b == NULL) {
			return a;
		}
		Node* c;
		if(a->data < b->data) {
			c  = a;
			c->next = merge(a->next, b);
		}
		else {
			c = b;
			c->next = merge(a, b->next);
		}
		return c;
	}
	Node* mergeSort(Node* a) {
		if(a == NULL || a->next == NULL) {
			return a;
		}
		Node* mid = midPoint(a);
		Node* aa = a;
		Node* b = mid->next;
		mid->next = NULL;
		aa = mergeSort(a);
		b = mergeSort(b);

		Node* c = merge(aa, b);
		return c;
	}
###### Given a linked list of 0s, 1s and 2s, sort it 
	Node* segregate(Node *head) {
		Node* temp=head;
		  int m[3]={0};                                 //to store count of 0 ,1 and 2 
		while(temp!=NULL)
		{
			m[temp->data]++;
			temp=temp->next;
		}
		temp=head;
		int i=0;
		while(temp!=NULL)
		{
			while(m[i]!=0)                               //till count does not become zero of a number(0,1 and 2)

			{
				temp->data=i;                           //keep assigning to data of node
				m[i]--;
				temp=temp->next;
			}
			i++;
		}
		return head;
	}
###### Merge Sort on Doubly Linked List
	struct node *splitList(struct node *head)
	{
		node* slow=head,*fast=head;
		while(1){
			fast = fast->next;
			if(fast==NULL||fast->next==NULL){
				node* t=slow->next;
				slow->next=NULL;
				t->prev=NULL;
				return t;
			} 
			fast=fast->next;
			slow = slow->next;
		}
	}
	struct node *merge(struct node *first, struct node *second){
		// Code here
		if(first==NULL){return second;}
		if(second==NULL) return first;
		if(first->data<second->data){
			first->prev=NULL;
			first->next=merge(first->next,second);
			if(first->next) first->next->prev=first;
			return first;
		} else{
			second->prev=NULL;
			second->next=merge(first,second->next);
			if(second->next) second->next->prev=second;
			return second;
		}
	}
###### Merge K sorted linked lists

######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
######
