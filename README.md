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
	Node *merge(Node *p,Node *q)
	{
	    if(p==NULL)
	    return q;
	    if(q==NULL)
	    return p;
	    Node *head;
	    if(p->data<q->data)
	    {
		head=p;
		head->next=merge(p->next,q);
	    }
	    else
	    {
		head=q;
		head->next=merge(q->next,p);
	    }
	    return head;
	}
	Node * mergeKList(Node *arr[], int N)
	{
	    int n=N;
	    if(N<2)
	    {
		if(n==1)
		return arr[0];
		else 
		return NULL;
	    }
	    Node *head=merge(arr[0],arr[1]);
	    for(int i=2;i<n;i++)
	    {
		head=merge(head,arr[i]);
	    }
	    return head;
	}
###### Intersection Point in Y Shapped Linked Lists
	int intersectPoint(Node* head1, Node* head2)
	{
	    unordered_set<Node*>set;
	    while(head1->next != NULL)
	    {
		set.insert(head1);
		head1 = head1->next;
	    }

	    int flag = 0;
	    while(head2->next != NULL)
	    {
		if(set.find(head2) != set.end())
		{
		    flag = 1;
		    return head2->data;
		}
		head2 = head2->next;
	    }
	    return -1;
	}
###### Clone a linked list with next and random pointer
	Node * copyList(Node *head)
	{
	     // Your code here
	     map <Node*, Node*> hash;
	     Node *p = head;
	     Node *head1 = create(p->data);
	     Node *temp = head1;
	     hash.insert(pair <Node*, Node*> (p, head1));
	     p=p->next;
	     while(p)
	     {
		 Node *q = create(p->data);
		 hash.insert(pair <Node*, Node*> (p, q));
		 temp->next = q;
		 temp = temp->next;
		 p = p->next;
	     }
	     p = head;
	     Node *q = head1;
	     while(p){
		 q->arb = hash[p->arb];
		 q=q->next;
		 p=p->next;
	     }
	     return head1;

	}
###### Add two numbers represented by Linked List
	Node* addSameSize(Node* head1, Node* head2, int* carry) {
	    if (head1 == nullptr) return nullptr; 
	    Node* result =  new Node(); 
	    result->next = addSameSize(head1->next, head2->next, carry);
	    int sum = head1->data + head2->data + *carry; 
	    *carry = sum / 10; 
	    sum %= 10; 
	    result->data = sum;
	    return result; 
	} 
	void addCarryToRemaining(Node* head1, Node* curr, int* carry, Node** result) { 
	    if (head1 == curr) return;

	    addCarryToRemaining(head1->next, curr, carry, result);
	    int sum = head1->data + *carry;
	    *carry = sum / 10;
	    sum %= 10; 
	    push(result, sum); 
	} 
###### LRU Cache
    int get(int key)
    {
        if(mp.find(key) == mp.end())
            return -1;
            
        auto it = mp[key];
        int val = it->second;
        lst.erase(it);
        lst.push_front({key, val});
        mp[key] = lst.begin();
        
        return val;
    }   
    void set(int key, int value)
    {
        if(mp.find(key) == mp.end()){
            if(lst.size() == size){
                auto last = lst.back();
                mp.erase(last.first);
                lst.pop_back();
            }
        }
        else{
            auto it = mp[key];
            lst.erase(it);
        }
          
        lst.push_front({key, value});
        mp[key] = lst.begin();
    }
## Stack

###### Removing consecutive duplicates
	string removeConsecutiveDuplicates(string s)
	{
		stack<int> myStack;
		string res="";
		for( int i=0;i <s.length();i++){
			if ( myStack.empty() || myStack.top() != s[i]){
				myStack.push(s[i]);
				res+=s[i];
			}         
		}
		return res;
	}
###### Removing consecutive duplicates - 2
	string removePair(string str){
	   stack<char> myStack;
	   for( int i=0; i<str.length();i++){
		   if ( myStack.empty() || myStack.top() != str[i])
			   myStack.push(str[i]);
			else{
				myStack.pop();
			}
		}
	   string res="";
	   while( myStack.empty() == false){
		   res=myStack.top()+ res;
		   myStack.pop();
	   }
	   return res;
	}
###### Implement two stacks in an array
	void twoStacks :: push1(int x)
	   {
		   if(abs(top1-top2)>=1){
			  top1++;
			  arr[top1] = x;
		   }
	   }
	void twoStacks ::push2(int x)
	   {
		   if(abs(top1-top2)>=1){
			   top2--;
			   arr[top2] = x;
		   }
	   }
	int twoStacks ::pop1()
	   {
		   int x = -1;
		   if(top1 >=0){
			  x = arr[top1];
			 top1--;
		   }
		   return x;
	   }
	int twoStacks :: pop2()
	   {
		   int x = -1;
		   if(top2 <size){
			  x = arr[top2];
			 top2++;
		   }
		   return x;
	   }
###### Get min at pop
	stack<int>_push(int arr[],int n)
	{
	   stack<int> s;
	   int m=INT_MAX;
	   for(int i=0;i<n;i++){
		   s.push(arr[i]);
		   m=min(arr[i],m);
		   s.push(m);
	   }
	   return s;
	   // your code here
	}
	void _getMinAtPop(stack<int>s)
	{
		while(!s.empty()){
			cout<<s.top()<<" ";
			s.pop();
			s.pop();
		}
		// your code here
	}
###### Delete middle element of a stack
	stack<int> deleteMid(stack<int>s,int sizeOfStack,int current)
	{
		stack<int>q;int i;
		int a=s.size()/2;
		for(i=0;i<a;i++)
		{q.push(s.top());//cout<<" a"<<q.top();
		s.pop();}s.pop();
		while(!q.empty()){s.push(q.top());
		q.pop();}
		return s;
	}
###### Stock span problem
	vector <int> calculateSpan(int price[], int n)
	{
	   vector<int> dp(n+1);
	   stack<int> s;
	   for(int i=0;i<n;i++)
	   dp[i]=1;
	   for(int i=0;i<n;i++)
	   {
		   if(s.empty())
		   s.push(i);
		   else
		   {
			   if(price[i]<price[s.top()])
			   s.push(i);
			   else
			   {
				   while(!s.empty()&&price[s.top()]<=price[i])
				   {
					   dp[i]+=dp[s.top()];
					   s.pop();
				   }
				   s.push(i);
			   }
		   }
	   }
	   return dp;
	}
###### Next larger element
	vector <long long> nextLargerElement(long long arr[], int n){
		vector<long long> res(n,-1);
		stack<int> s;
		for(int i=0;i<n;i++)
		{
			if(s.empty())
			{
				s.push(i);
			}   
			else
			{   
				while(!s.empty()&&arr[s.top()]<arr[i])
				{
					res[s.top()]=arr[i];
					s.pop();
				}      
				s.push(i);
			}
		}
		return res;
	}
###### Maximum Rectangular Area in a Histogram
	long getMaxArea(long long arr[], int n){
		stack<int> st; long area=0,max=0; int i=0;
		while(i<n){ if(st.empty()||arr[i]>=arr[st.top()]){
		st.push(i);
		i++;
		}
		else{
		int curr = st.top();st.pop();
		area = arr[curr]*(st.empty()?i:(i-1-st.top()));
		if(area>max)
		max=area;
		}
		}
		while(!st.empty()){
		int curr = st.top();st.pop();
		area = arr[curr]*(st.empty()?i:(i-1-st.top()));
		if(area>max)
		max=area;
		}
		return max;
	}
######  The Celebrity Problem
	int getId(int M[MAX][MAX], int n)
	{
		vector<int> v;
		int i,j,cnt;
		for(j=0;j<n;j++)
		{
			cnt=0;
			for(i=0;i<n;i++)
			{
				if(M[i][j]==1)
				cnt++;
			}
			if(cnt>=n-1)
			v.push_back(j);
		}
		return (v.size()==1) ? v[0] : -1;
	}
###### Maximum of minimum for every window size
	void printMaxOfMin(int arr[], int n) 
	{ 
		stack<int> s;   
		int left[n+1]; 
		int right[n+1];  
		for (int i=0; i<n; i++) 
		{ 
			left[i] = -1; 
			right[i] = n; 
		} 
		for (int i=0; i<n; i++) 
		{ 
			while (!s.empty() && arr[s.top()] >= arr[i]) 
				s.pop(); 

			if (!s.empty()) 
				left[i] = s.top(); 

			s.push(i); 
		} 
		while (!s.empty()) 
			s.pop();  
		for (int i = n-1 ; i>=0 ; i-- ) 
		{ 
			while (!s.empty() && arr[s.top()] >= arr[i]) 
				s.pop(); 

			if(!s.empty()) 
				right[i] = s.top(); 

			s.push(i); 
		} 
		int ans[n+1]; 
		for (int i=0; i<=n; i++) 
			ans[i] = 0; 
		for (int i=0; i<n; i++) 
		{ 
			int len = right[i] - left[i] - 1; 
			ans[len] = max(ans[len], arr[i]); 
		} 
		for (int i=n-1; i>=1; i--) 
			ans[i] = max(ans[i], ans[i+1]); 
		for (int i=1; i<=n; i++) 
			cout << ans[i] << " ";
		cout << endl;
	} 
## Queue

######  Stack using two queues
	void QueueStack :: push(int x)
	{
			q1.push(x);
	}
	int QueueStack :: pop()
	{
			int ans= -1;
			if(q1.empty()) return ans;
			q1.push(0);
			while(1)
			{
				int x= q1.front();
				q1.pop();
				if(q1.front()==0)
				{
					ans=x;
					break;
				}
				else 
				{q1.push(x);}
			}
			q1.pop();
			return ans;
	}
###### Maximum of all subarrays of size k
	void max_of_subarrays(int *arr, int n, int k){
		deque<int>Qi;
		for(int i=0;i<k;i++){
			 while ((!Qi.empty()) && arr[i] >= arr[Qi.back()]) 
				Qi.pop_back(); 
			Qi.push_back(i);
		}
		  for (int i=k; i < n; ++i) {  
			cout<<arr[Qi.front()]<<" "; 
			while ((!Qi.empty()) && Qi.front() <= i - k) 
				Qi.pop_front(); 
			while ((!Qi.empty()) && arr[i] >= arr[Qi.back()]) 
				Qi.pop_back(); 
			Qi.push_back(i);
		} 
		cout<< arr[Qi.front()]<<" ";
	}
###### Generate Binary Numbers
	vector<string> generate(ll n)
	{
		string s="1";
		queue<string> q;
		q.push(s);
		vector<string> res;
		while(n--)
		{
			string cur=q.front();
			q.pop();
			res.push_back(cur);

			q.push(cur+"0");
			q.push(cur+"1");
		}
		return res;
	}
###### Reverse First K elements of Queue
	{
		stack<ll> s;
		queue<ll> q1;
		while(k-->0)
		{
			s.push(q.front());
			q.pop();
		}
		while(!s.empty())
		{
			q1.push(s.top());
			s.pop();
		}
		while(!q.empty())
		{
			q1.push(q.front());
			q.pop();
		}
		return q1;
	}
## Tree

###### Determine if Two Trees are Identical
	bool isIdentical(Node *r1, Node *r2)
	{
		if(r1==NULL&&r2==NULL)
		{return true;}
		else if(r1==NULL||r2==NULL)
		{return false;}
		else
		{
			if(r1->data!=r2->data)
			{return false;}
			else
			{
				return isIdentical(r1->left,r2->left)&&isIdentical(r1->right,r2->right);
			}
		}
	}

###### Children Sum Parent
	int isSumProperty(Node *root)
	{
		 if(root==NULL || (root->left==NULL && root->right==NULL))
			return 1;
		 int l=0,r=0;
		 if(root->left)
		 l=root->left->data;
		 if(root->right)
		 r=root->right->data;
		 return ((root->data==l+r) && isSumProperty(root->left) && isSumProperty(root->right));
	}
###### Level order traversal Line by Line
	void levelOrder(Node* node)
	{
	 queue<Node*> q;
	 q.push(node);
	 while(1)
	 {
		 int size = q.size();
		 if(size == 0)
			break;
		 while(size>0)
		 {
			 node = q.front();
			 q.pop();
			 cout<<node->data<<" ";
			 if(node->left)
			 q.push(node->left);
			 if(node->right)
			 q.push(node->right);
			 size--;
		 }
		 cout<<"$ ";
	 }
	}
###### Level order traversal in spiral form
	void printSpiral(Node *root)
	{
		if(root==NULL)
		return;
		queue<Node*>q;
		q.push(root);
		int n=false;
		while(!q.empty()){
			int size=q.size();
			vector<int>v(size);
			for(int i=0;i<size;i++){
				Node* temp=q.front();
				q.pop();
				int index=(n)?i:(size-1-i);
				v[index]=temp->data;
				if(temp->left)
				q.push(temp->left);
				if(temp->right)
				q.push(temp->right);
			}
			n=!n;
			for(auto it=v.begin();it!=v.end();it++)
			cout<<*it<<" ";
			v.clear();
		}
	}
###### Maximum Width of Tree
	int getMaxWidth(Node* root)
	{
	   int maxwidth = 0, ans = 1;
	   queue<Node*> q;
	   if(!root) return 0;
	   q.push(root);
	   int n = 0;
	   while(!q.empty()) {
		   if(maxwidth >= q.size())
				maxwidth = 0;
		   Node* top = q.front();
		   q.pop();
		   if(top->left) {
			   maxwidth++;
			   q.push(top->left);
		   }
		   if(top->right) {
			   maxwidth++;
			   q.push(top->right);
		   }
		   ans = max(ans, maxwidth);
	   }
	   return ans;
	}
###### Check for Balanced Tree
	int height(Node* root)
	{
		if(root==nullptr)
		return 0;
		return 1+max(height(root->left), height(root->right));
	}
	bool isBalanced(Node *root)
	{
		if(root==nullptr)
		return 1;
		int l = height(root->left);
		int r = height(root->right);
		if(abs(l-r)<=1 &&isBalanced(root->left) && isBalanced(root->right))
			return 1;
		return 0;
	}
###### Left View of Binary Tree
	int mxlevel=-1;
	void printLeftView(Node *root,int level){
		if(root == NULL) return;
		if(level > mxlevel){
			mxlevel = level;
			cout<<root->data<<" ";
		}
		printLeftView(root->left,level+1);
		printLeftView(root->right,level+1);
	}
	void leftView(Node *root){
	   mxlevel = -1;
	   if(root==NULL) return;
	   printLeftView(root,0);
	}
###### Right View of Binary Tree
	void rightView(Node *root)
	{
		if(root==NULL)
			return ;
		queue<Node*>Q;
		Q.push(root);
		Q.push(NULL);
		while(Q.front()!=NULL){
			Node *x=Q.front();
			Q.pop();
			while(x!=NULL){
				if(x->left)
					Q.push(x->left);
				if(x->right)
					Q.push(x->right);
				if(Q.front()==NULL)
					cout<<x->data<<" ";
				x=Q.front();
				Q.pop();
			}
			Q.push(NULL);
		}
	}
###### Lowest Common Ancestor in a Binary Tree
	Node * lca(Node* root ,int n1 ,int n2 )
	{ 
		if(root==NULL)
			return NULL;
		if(root->data==n1)
			return root;
		if(root->data==n2)
			return root;
		Node *l=lca(root->left,n1,n2);
		Node *r=lca(root->right,n1,n2);
		if(l && r) return root;
		if(l==NULL && r==NULL) return l;
		if(l) return l;
		if(r) return r;
	}
###### Diameter of Binary Tree
	int d = 0;
	int compute(Node* node){
		if(node == NULL)return 0;
		if(node->left == NULL && node -> right == NULL) return 1;
		int leftHeight = compute(node->left);
		int rightHeight = compute(node->right);
		d = max(d,leftHeight+rightHeight+1);
		return (1+max(leftHeight,rightHeight));
	}
	int diameter(Node* node)
	{
	   d = 0;
	   compute(node);
	   return d;
	}
###### Vertical Width of a Binary Tree
	int l;
	int r;
	void func(Node* root,int pos)
	{
		if(root == NULL)
			return;

		if(pos < 0)
			l = max(l,abs(pos));
		else
			r = max(r,pos);

		func(root->left,pos-1);
		func(root->right,pos+1);
	}

	int verticalWidth(Node* root)
	{
		l = 0;
		r = 0;
		func(root,0);
		return l+r+1;
	}
###### Mirror Tree
	void mirror(Node* node) 
	{
		 if(node == NULL){
			 return;
		 }
		 else{
			 mirror(node->left);
			 mirror(node->right);

			 Node *temp = node->left;
			 node->left = node->right;
			 node->right = temp;
		 }
	}
###### Check if subtree
	bool isidentical(Node *t1,Node *t2)
	{
		if(t1==NULL&&t2==NULL)
		return true;
		if(t1==NULL || t2==NULL) return false;
		bool l=isidentical(t1->left,t2->left);
		bool r=isidentical(t1->right,t2->right);
		if(t1->key==t2->key&&l&&r)
		{
		 return true;   
		}
		else
		return false;
	}
	bool isSubtree(Node*  T1,Node * T2)
	{
		if(T1==NULL&&T2==NULL)
		return true;
		if(T1==NULL || T2==NULL) return false;
		if(T1->key==T2->key)
		{
			if(isidentical(T1,T2)==true)
			return true;
			else
			{
				bool l=isSubtree(T1->left,T2);
				bool r=isSubtree(T1->right,T2);
				return l||r;
			}
		}
		bool l=isSubtree(T1->left,T2);
		bool r=isSubtree(T1->right,T2);
		return l||r;

	}
###### Make Binary Tree From Linked List
	void convert(node *head,TreeNode * &root)
	{
		if(head==NULL)
		return;
		root=newnode(head->data);
		head=head->next;
		queue<TreeNode *> q;
		q.push(root);
		while(head)
		{
			TreeNode *t=q.front();
			q.pop();
			t->left=newnode(head->data);
			q.push(t->left);
			if(head)
			head=head->next;
			if(head)
			{
				t->right=newnode(head->data);
				q.push(t->right);
				head=head->next;
			}
		}
	}
###### Binary Tree to DLL
	void bToDLL(Node *root, Node **head)
	{
		if(root==NULL) return;
		static Node* prev= NULL;
		bToDLL(root->left,head);
		if(*head==NULL)
		{
			prev=NULL;
			*head = root;
		}
		else
		{
			root->left = prev;
			prev->right=root;
		}
		prev = root;
		bToDLL(root->right,head);
	}
######  Binary Tree to CDLL 
	Node *bTreeToCList(Node *root)
	{
		Node *cur = root;
		Node *head = NULL, *last;
		stack<Node *> st; 
		while(cur || !st.empty()) {
			while(cur){
				st.push(cur);
				cur = cur->left;
			}
			cur = st.top();
			st.pop();
			Node *temp = newNode(cur->data);
			if(!head){
				head = temp;
				last = head;
			}
			else{
				temp->left = last;
				last->right = temp;
				last = temp;
			}
			cur = cur->right;
		}
		last->right = head;
		head->left = last;
		return head;
	}
###### Connect Nodes at Same Level
	void connect(Node *p){
	   queue< Node * > qq;
	   qq.push(p);
	   while(!qq.empty()){
		   int sz = qq.size();
		   while(sz > 0){
			   Node* node = qq.front();
			   qq.pop();
			   if(sz > 1){
				   node->nextRight = qq.front();
			   }else{
				   node->nextRight = NULL;
			   }
			   if(node->left != NULL)
			   qq.push(node->left);
			   if(node->right != NULL)
			   qq.push(node->right);
			   sz--;
		   }
	   }
	}
###### Construct Binary Tree from Parent Array
	Node* createTree(int parent[], int n)
	{
		Node* root;
		Node* a[n];
		for(int i=0;i<n;i++){
			a[i] = (Node*)malloc(sizeof(Node));
			a[i]->data = i;
		}
		bool hasLeft[n];
		for(int i=0;i<n;i++)
			hasLeft[i] = false;
		for(int i=0;i<n;i++){
			int p=parent[i];
			if(parent[i] == -1) root = a[i];
			else{
				if(hasLeft[p]){
					a[p]->right = a[i];
				}
				else{
					a[p]->left = a[i];
					hasLeft[p] = true;
				}
			}
		}
		return root;
	}
###### Tree from Postorder and Inorder
	Node *tree(int in[],int post[],int l,int r)
	{
		if(l>r)
		{
			return NULL;
		}
		Node *q=newnode(post[x]);
		x--;
		if(l==r)
		{
			return q;
		}
		int k;
		for(int i=l;i<=r;i++)
		{
			if(in[i]==q->data)
			{
				k=i;
			}
		}
		q->right=tree(in,post,k+1,r);
		q->left=tree(in,post,l,k-1);
		return q;
	}
	Node *buildTree(int in[], int post[], int n)
	{
		x=n-1;
		Node *p=tree(in,post,0,n-1);
		return p;
	}
###### Foldable Binary Tree
	bool check(node *p1, node *p2)
	{
		if(p1 == NULL && p2 == NULL)
			return true;

		if(p1 == NULL || p2 == NULL)
			return false;

		bool l,r;
		l = check(p1->left,p2->right);
		r = check(p1->right,p2->left);

		return l and r;
	}

	bool isFoldable(struct node *root)
	{
		return check(root,root);
	}
###### Maximum path sum from any node
	int findMaxUtil(Node* root, int &res)
	{
		if(root)
		{
			int l=findMaxUtil(root->left,res);
			int r=findMaxUtil(root->right,res);
			int p=root->data;
			int mx=max(max(l,r)+p,p);
			res=max(res,max(mx,l+r+p));
			return mx;
		}
		return 0;
	}
###### Maximum difference between node and its ancestor
	int ans = -1;
	int find(Node* root){
		if(root == NULL) return INT_MAX;
		if(root->left == NULL && root-> right == NULL) return root->data;
		int lMin = find(root->left);
		int rMin = find(root->right);
		ans = max(ans,root->data - min(lMin,rMin));
		return min(root->data,min(lMin,rMin));
	}
	int maxDiff(Node* root)
	{
		ans = -1;
		find(root);
		return ans;
	}
###### Count Number of SubTrees having given Sum
	int sum(Node *root){
		if(root==NULL){
			return 0 ;
		}
		return root->data+sum(root->left)+sum(root->right) ;
	}
	int countSubtreesWithSumX(Node* root, int x)
	{
		if (!root)return 0;
		if(sum(root) == x){
			return 1+countSubtreesWithSumX(root->left , x)+countSubtreesWithSumX(root->right , x);   
		}
		else{
			return countSubtreesWithSumX(root->left , x)+countSubtreesWithSumX(root->right , x) ;
		}	
	}
###### Serialize and Deserialize a Binary Tree
	void serialize(Node *root,vector<int> &A)
	{
		if(root == NULL)
			A.push_back(-1);
		else
		{
			A.push_back(root->data);
			serialize(root->left,A);
			serialize(root->right,A);
		}
	}
	struct Node *newnode(int d)
	{
		struct Node *add = new Node;
		add->left = NULL;
		add->right = NULL;
		add->data = d;
		return add;
	}
	Node * deSerialize(vector<int> &a)
	{
		struct Node *k;
		if(!a.empty())
		{
			if(a[0] == -1)
			{
				a.erase(a.begin()+0);
				return NULL;
			}
			else
			{
				k = newnode(a[0]);
				a.erase(a.begin()+0);
				k->left = deSerialize(a);
				k->right = deSerialize(a);
				return k;
			}
		}
	}
###### Node at distance
	void kDistantFromLeafUtil(Node* node, int path[], bool visited[], int pathLen, int k)
	{
		if(node)
		{
			path[pathLen]=node->key;
			visited[pathLen]=false;
			if(!node->left&&!node->right&&(pathLen-k)>=0)
			{
				if(visited[pathLen-k]==false)
				{
					counter++;
					visited[pathLen-k]=true;
				}
			}
			kDistantFromLeafUtil(node->left,path,visited,pathLen+1,k);
			kDistantFromLeafUtil(node->right,path,visited,pathLen+1,k);
		}
	}
###### ZigZag Tree Traversal
	vector <int> zigZagTraversal(Node* root)
	{
		queue<Node*>q;
		q.push(root);
		vector<int>res;
		int n=true;
		while(!q.empty()){
			int size=q.size();
			vector<int>row(size);
			int index;
			for(int i=0;i<size;i++){
				Node* temp=q.front();
				q.pop();
				index=(n)?i:(size-i-1);
				row[index]=temp->data;
				if(temp->left)
				q.push(temp->left);
				if(temp->right)
				q.push(temp->right);
			}
			std::copy(row.begin(),row.end(),std::back_inserter(res));
			row.clear();
			n=!n;
		}
		return res;
	}
###### Maximum sum of Non-adjacent nodes
	int getMaxSum(Node *root) 
	{

		if(root==NULL) return 0;
		int include=0,exclude=0;
		if(root->left) include+= getMaxSum(root->left->left)+getMaxSum(root->left->right);
		if(root->right) include+= getMaxSum(root->right->left)+getMaxSum(root->right->right);
		include+=root->data;

		exclude+=getMaxSum(root->left)+getMaxSum(root->right);
		return max(include,exclude);
	}
## Binary Search Tree

###### Check for BST
	bool checkBST(Node* root,int mn,int mx){
		if(root == NULL) return true;
		if(root->data <= mn || root->data >= mx)return false;
		return checkBST(root->left,mn,root->data) && checkBST(root->right,root->data,mx);
	}
	bool isBST(Node* root) {
		return checkBST(root,INT_MIN,INT_MAX);
	}
###### Minimum element in BST
	int minValue(struct node* root)
	{
		if(root->left==NULL)
		{
			return root->data;
		}
		while(root->left!=NULL)
		{
			root=root->left;
		}
		minValue(root);
	}
###### Print Common Nodes in two BSTs
	void getValues(Node *root, set<int> *s)
	{
		if(root)
		{
			s->insert(root->data);
			getValues(root->left,s);
			getValues(root->right,s);
		}
	}
	void printCommon(Node *root1, Node *root2)
	{
		 set<int> s1;
		 set<int> s2;  
		 getValues(root1,&s1);
		 getValues(root2,&s2);
		 set<int> :: iterator it;
		 for(it=s1.begin();it!=s1.end();it++)
		 {
			 set<int> :: iterator it2;
			 it2=s2.find(*it);
			 if(it2!=s2.end())
			 {
				 cout<<*it<<" ";
			 }
		 }
		 cout<<endl;
	}
###### Lowest Common Ancestor in a BST
	Node* LCA(Node *root, int n1, int n2)
	{
		if(root==NULL) return NULL;
		if((n1<root->data)&&(n2<root->data))
		return LCA(root->left,n1,n2);
		if((n1>root->data)&&(n2>root->data))
		return LCA(root->right,n1,n2);
		return root;
	}
###### Print BST elements in given range
	void printNearNodes(Node *root, int l, int h)
	{   
	  if(root==NULL)
	  {return;}
	  printNearNodes(root->left,l,h);
	  if(root->data>=l&&root->data<=h)
			cout<<root->data<<" ";
	  printNearNodes(root->right,l,h);
	}
###### Pair Sum in BST
	unordered_map<int,bool> m;
	bool Pair(Node* root, int sum) {
		if(root == NULL)
			return false;
		bool l,r;
		l = Pair(root->left,sum);
		if(m[sum-root->data])
			return true;
		m[root->data] = true;
		r = Pair(root->right,sum);
		return l or r; 
	}

	bool findPair(Node* root, int sum) {
		m.clear();
		return Pair(root,sum);
	}
###### Floor in BST
	int floor(Node* root, int key) 
	{ 
		if(!root) 
			return INT_MAX; 

		if(key == root->data)
			return key;

		if(key < root->data)
			return floor(root->left,key);

		int x = floor(root->right,key);

		if(x <= key)
			return x;
		else
			return root->data;       
	}
###### Ceil in BST
	int findCeil(Node* root, int input) 
	{ 
		if (root == NULL) 
			return -1; 
		if(input == root->data)   
			return input;
		if(input > root->data)
			return findCeil(root->right,input);
		int x = findCeil(root->left,input);
		if(x >= input)
		   return x;
		else
			return root->data;
	}
###### Vertical Traversal of Binary Tree
	map<int,vector<int>> mp;
	void preorder(Node *root,int level){
		if(root == NULL) return;
		mp[level].push_back(root->data);
		preorder(root->left,level-1);
		preorder(root->right,level+1);
	}

	void verticalOrder(Node *root){
		map<int,vector<int>>::iterator it;
		mp.clear();
		preorder(root,0);
		for(it=mp.begin();it!= mp.end();it++){
			vector<int> v = it->second;
			for(int i=0;i<v.size();i++){
				cout<<v[i]<<" ";
			}
		}
	}
###### Top View of Binary Tree
	void topView(struct Node *root)
	{       map<int,int>mp;
			queue<pair<Node*,int>>q;
			int hd=0;
			if(root==NULL)
				return;
			q.push(make_pair(root,hd));
			while(!q.empty())
			{   pair<Node*,int> no=q.front();
					q.pop();

				int val=no.second;
				struct Node* nod=no.first;
				if(mp.find(val)==mp.end())
					mp[val]=nod->data;
				if(nod->left!=NULL)
					q.push(make_pair(nod->left,val-1));
				if(nod->right!=NULL)
					q.push(make_pair(nod->right,val+1));
			}
			for(auto x:mp)
				cout<<x.second<<" ";
	}

###### Bottom View of Binary Tree
	map<int,int> mp;
	void preorder(Node *root,int haxis){
		if(root==NULL) return;
		mp[haxis] = root->data;
		preorder(root->left,haxis-1);
		preorder(root->right,haxis+1);
	}
	void bottomView(Node *root){
		 mp.clear();
		 preorder(root,0);
		 map<int,int>::iterator it;
		 for(it= mp.begin();it!=mp.end();it++){
			 cout<<it->second<<" ";
		 }
	}
###### Find the Closest Element in BST
	int minDiff(Node *root, int k)
	{
		int m=INT_MAX;
		queue<Node*> q;
		q.push(root);
		while(q.size()){
			if(abs(k-q.front()->data)<m){
				m=abs(k-q.front()->data);
			}
			if(q.front()->right) q.push(q.front()->right);
			if(q.front()->left) q.push(q.front()->left);
			q.pop();
		}
		return m;
	}
###### Convert Level Order Traversal to BST
	Node *BSTutil(Node *root, int data)
	{
		if(root==NULL)
		{
			root=new Node(data);
			return root;
		}
		if(root->data>data)
			root->left=BSTutil(root->left,data);
		else
			root->right = BSTutil(root->right,data);
	return root;   
	}
	Node* constructBst(int arr[], int n)
	{
		int i;
		if(n==0) return NULL;
		Node *root=NULL;
		for(int i=0;i<n;i++)
			root= BSTutil(root,arr[i]);
		return root;	
	}
###### Count BST nodes that lie in a given range
	int getCountOfNode(Node *root, int l, int h)
	{ 
		int count=0;
		if(root==NULL)
		return count ;
		if(root->data>=l && root->data<=h)
		count++;
		return count+ getCountOfNode(root->left, l, h)+getCountOfNode(root->right, l, h);
	}
###### Merge two BST 's
	vector<int> v;
	void inorder(Node* root)
	{
		if(root == NULL)
			return;

		inorder(root->left);
		v.push_back(root->data);
		inorder(root->right);
	}

	void merge(Node *root1, Node *root2)
	{
		v.clear();
		inorder(root1);
		vector<int> x = v; 
		v.clear();
		inorder(root2);
		vector<int> y = v;

		int f = 0;
		int s = 0;
		while(f<x.size() && s<y.size())
		{
			if(x[f] <= y[s])
			{
				cout<<x[f]<<" ";
				f++;
			}
			else
			{
				cout<<y[s]<<" ";
				s++;
			}
		}
		while(f < x.size())
		{
			cout<<x[f]<<" ";
			f++;
		}
		while(s < y.size())
		{
			cout<<y[s]<<" ";
			s++;
		}
	}
###### Smaller on Right
int countSmallerRight(int a[], int n) {
    int mx=0;
    set<int>s;
    for(int i=n-1;i>=0;i--){
        s.insert(a[i]);
        auto l=s.lower_bound(a[i]);
        int p=distance(s.begin(),l);
        mx=max(mx,p);
        
    }
    return mx;
}
###### Preorder to Postorder
	void findPostOrderUtil(int pre[],int n, int minval, int maxval, int &pre_index)
	{
		if(pre_index==n)
		{
			return;
		}
		if(pre[pre_index]<minval||pre[pre_index]>maxval)
		{
			return;
		}
		int val = pre[pre_index];
		pre_index++;
		findPostOrderUtil(pre,n,minval,val,pre_index);
		findPostOrderUtil(pre,n,val,maxval,pre_index);
		cout<<val<<" ";
	}
	void findPostOrder(int arr[],int n)
	{
		int pre_index = 0;
		findPostOrderUtil(arr,n,INT_MIN,INT_MAX,pre_index);
	}
###### Fixing Two nodes of a BST
	vector<node *> v;
	void inorder(node *root)
	{
		if(root)
		{
			inorder(root->left);
			v.push_back(root);
			inorder(root->right);
		}
	}
	struct node *correctBST( struct node* root )
	{
		v.clear();node *p,*q;int k;
		inorder(root);
		for(int i=0;i<v.size()-1;i++)
		{
			if(v[i]->data>v[i+1]->data)
			{
				p=v[i];k=i;
				break;
			}
		}
		for(int i=k+1;i<v.size()-1;i++)
		{
			if(v[i]->data>v[i+1]->data)
			{
				q=v[i+1];
				break;
			}
		}
		int temp=p->data;
		p->data=q->data;
		q->data=temp;
		return root;
	}
## Heap

######  K largest elements 
	vector<int> kLargest(int arr[], int n, int k)
	{
		vector<int>res;
		priority_queue<int>pq;
		for(int i=0;i<n;i++)
		{
			pq.push(arr[i]);
		}
		for(int i=0;i<k;i++)
		{
		   res.push_back(pq.top());
		   pq.pop();
		}
		return res;
	}
######  Kth largest element in a stream

	void kthLargest(int arr[], int n, int k)
	{
		vector<int>res;
		priority_queue<int,vector<int>,greater<int>>pq;
		queue<int>q;
		for(int i=0;i<n;i++)
		{
			if(pq.size() < k) 
				pq.push(arr[i]);
			else // if size becomes equal to k
			{
				if(arr[i] > pq.top()) // if top element is smaller than arr[i]
				{
					pq.pop();
					pq.push(arr[i]);
				}
			}

			if(pq.size()<k)
				cout<<-1<<" ";
			else
				cout<<pq.top()<<" "; // print the current top element
		}
	}

###### K Most occurring elements
	int print_N_mostFrequentNumber(int arr[],int n, int k) 
	{               
			unordered_map<int,int>um;
			for(int i=0;i<n;i++)
				um[arr[i]]++;
			int sum=0;
			priority_queue<int>pq;
			for(auto &y:um)
			{   pq.push(y.second);
			}
			int i=0;
			while(!pq.empty()&&i<k)
			{   sum=sum+pq.top();
					pq.pop();
					i++;
			}
			return sum; 
	} 
###### Minimum Cost of ropes
	long long minCost(long long arr[], long long n) {
			priority_queue<long long,vector<long long>,greater<long long>>pq;
			long long i,sum1=0;
			for(i=0;i<n;i++)
			{    pq.push(arr[i]);sum1=sum1+arr[i];}
			long long cost=0,sum=0;
			while(sum1!=pq.top())
			{   sum=0;
				sum=sum+pq.top();
				pq.pop();
				sum=sum+pq.top();
				pq.pop();
				cost=cost+sum;
				pq.push(sum);
			}
			return cost;
	}
###### Nearly sorted
	vector <int> nearlySorted(int arr[], int num, int K){
			priority_queue<int,vector<int>,greater<int>>pq;
			for(int i=0;i<num;i++)
				pq.push(arr[i]);
			vector<int>v;
			while(!pq.empty())
			{   v.push_back(pq.top());
					pq.pop();
			}
			return v;
	}
###### Merge k Sorted Arrays
	int *mergeKArrays(int arr[][N], int k)
	{           
			priority_queue<int,vector<int>,greater<int>>pq;
			int *a=new int[k*k];
			int l=0;
			for(int i=0;i<k;i++)
			{   for(int j=0;j<k;j++)
				{   pq.push(arr[i][j]);
				if(pq.size()>(k-1)*k)
				{  a[l]=pq.top();
					pq.pop();l++;
				}}
			}
			while(pq.size()>0)
			{   a[l]=pq.top();
				pq.pop();
				l++;
			}
			return a;
	}
###### Rearrange characters
	string rearrangeString(string str){
			priority_queue<pair<int,char>>p;
		map<char,int>m;
		for(int i=0;i<str.length();i++){
			m[str[i]]++;
		}
		for(auto x:m){
			p.push(make_pair(x.second,x.first));
		}
		string res;
		pair<int,char>pre={-1,'#'};
		while(!p.empty())
		{
			auto cu=p.top();
			//you haven't popped anything from p, therefore add p.pop();
			p.pop();
			res.push_back(cu.second);
			cu.first-=1;
			if(pre.first>0)
				p.push(pre);
			pre=cu;
		}
		if(res.length()==str.length()){
			return res;
		}
		return "aa";
	}
###### Find median in a stream
	#include <bits/stdc++.h>
	using namespace std;

	class FindMedian
	{
		public:
			void insertHeap(int &);
			double getMedian();
		private:
			double median; //Stores current median
			priority_queue<int> max; //Max heap for lower values
			priority_queue<int, vector<int>, greater<int> > min; //Min heap for greater values
			void balanceHeaps(); //Method used by insertHeap
	};


	 // } Driver Code Ends


	// Function to insert heap

	void FindMedian::insertHeap(int &x)
	{
		if(max.size()==0)
		{
			max.push(x);
			getMedian();
		}
		else if(x>max.top())
		{
			min.push(x);
			balanceHeaps();
			getMedian();
		}
		else
		{max.push(x);balanceHeaps();getMedian();}

	}

	// Function to balance heaps
	void FindMedian::balanceHeaps()
	{
		if((max.size()-min.size())==2)
		{
			min.push(max.top());
			max.pop();
		}
		else if((min.size()-max.size())==1)
		{
			max.push(min.top());
			min.pop();
		}

	}

	// Function to return getMedian
	double FindMedian::getMedian()
	{
		if(max.size()-min.size()==1)
		return(max.top());
		else
		return((max.top()+min.top())/2);
	}

	// { Driver Code Starts.

	int main()
	{
		int n, x;
		FindMedian Ans;
		cin >> n;
		for(int i = 1;i<= n; ++i)
		{
			cin >> x;
			Ans.insertHeap(x);
			cout << floor(Ans.getMedian()) << endl;
		}
		// }
		return 0;
	}  // } Driver Code Ends
## Graph

###### Find the number of islands
	bool isValid{
		if((i>=0)&&(i<n)&&(j>=0)&&(j<m)&&(A[i][j]==1)&&(visited[i][j]==false))
		{
			return true;
		}
		else
		return false;
	}
	void dfs(int A[MAX][MAX],int i,int j,bool **visited,int n,int m)
	{
		/*if(!isvalid(A,i,j,visited,n,m))
		{
			return;
		}*/
		 static int r[] = {-1, -1, -1,  0, 0,  1, 1, 1};
		 static int c[] = {-1,  0,  1, -1, 1, -1, 0, 1};
		 visited[i][j]=true;
		 for(int k=0;k<8;k++)
		 {
			 if(isvalid(A,i+r[k],j+c[k],visited,n,m))
			 {
				 dfs(A,i+r[k],j+c[k],visited,n,m);
			 }
		 }

	}
	int findIslands(int A[MAX][MAX], int N, int M)
	{
		int count=0;
		bool **visited;
		visited=new bool*[N];
		for(int i=0;i<N;i++)
		{
			visited[i]=new bool[M];
		}
		for(int i=0;i<N;i++)
		{
			for(int j=0;j<M;j++)
			{
				visited[i][j]=false;
			}
		}
		for(int i=0;i<N;i++)
		{
			for(int j=0;j<M;j++)
			{
				if(!visited[i][j]&&A[i][j])
				{
					count++;
					dfs(A,i,j,visited,N,M);
				}
			}
		}
		return count;
	}
###### Find whether path exist
	bool safe(int x,int y,int n){
		return x>=0 && x<n && y>=0 && y<n;
	}
	void dfs(int x,int y,int n,int &ans){
		int row[]={-1,0,0,1};
		int col[]={0,-1,1,0};
		vis[x][y]=1;
		if(mat[x][y]==2){
			ans=1;
			return;
		}
		for(int k=0;k<4;k++){
			if(safe(x+row[k],y+col[k],n) && vis[x+row[k]][y+col[k]]==0 && mat[x+row[k]][y+col[k]]!=0 && ans==0){
				dfs(x+row[k],y+col[k],n,ans);
			}
		}
	}
###### Level of Nodes
	int Graph::levels( int s, int l){
		// Your code here
		bool visited[V];
		memset(visited,false,sizeof(visited));
		queue<pair<int,int>>q;
		q.push({s,0});
		visited[s]=true;
		int c = -1;
		while(!q.empty())
		{
			pair<int,int> p;
			p = q.front();
			q.pop();
			int f = p.first;
			int s = p.second;
			if(f==l)
			{
				c=s;
			}
			for(int i:adj[f])
			{
				if(!visited[i])
				{
					visited[i]=true;
					q.push({i,s+1});
				}
			}
		}
		return c;
	}
###### Possible paths between 2 vertices
	void DFSRec(list<int> adj[], int s, int d, bool visited[], int &count)
	{
		visited[s] = true;
		if(s == d)
		{
			count++;
		}
		else
		{
			for(int u: adj[s])
			{
				if(visited[u] == false)
				{
					DFSRec(adj, u, d, visited, count);

					visited[u] = false;
				}
			}
		}
	}

	int countPaths(list<int> adj[], int V, int s, int d)
	{
		bool visited[V+1] = {0};
		int count = 0;
		DFSRec(adj, s, d, visited, count);
		return count;
	}
###### X Total Shapes
	for(int j=0;j<m;j++)
	 {
		 if(a[i][j]=='X'&&v[i][j]==false)
		 {
			 count++;
			 dfs(i,j,a,v,n,m);
		 }
	 }
	 cout<<count<<"\n";
	 
	 bool isvalid(int i,int j,char **a,bool **v,int n,int m)
	{
		if(i<n&&i>=0&&j<m&&j>=0&&a[i][j]=='X'&&v[i][j]==false)
		return true;
		return false;
	}
	void dfs(int i,int j,char **a,bool **v,int n,int m)
	{
		if(v[i][j]==true)
		return;
		v[i][j]=true;
		static int r[]={0,1,0,-1};
		static int c[]={-1,0,1,0};
		for(int k=0;k<4;k++)
		{
			if(isvalid(i+r[k],j+c[k],a,v,n,m))
			{
				dfs(i+r[k],j+c[k],a,v,n,m);
			}
		}
	}
###### Distance of nearest cell having 1
	vector <vector <int> > nearest(vector<vector<int>> &mat, int N, int M) {
		  queue<pair<int,int>>q;
		  vector<vector<int>>dist(N,vector<int>(M,INT_MAX));
		  for(int i=0;i<mat.size();i++)
		  {
			  for(int j=0;j<mat[0].size();j++)
			  {
				  if(mat[i][j]==1)
				  {
				   q.push({i,j});
				   dist[i][j]=0;
				  }
			  }
		  }
		  int x_dir[]={-1,0,1,0};
		  int y_dir[]={0,-1,0,1};
		  while(!q.empty())
		  {
			  pair<int,int>temp=q.front();
			  int i=temp.first;
			  int j=temp.second;
			  q.pop();
			  for(int k=0;k<4;k++)
			  {
				  int x_new=i+x_dir[k];
				  int y_new=j+y_dir[k];
				  if(x_new>=0 && x_new<N && y_new>=0 && y_new<M && dist[x_new][y_new]>dist[i][j]+1)
				  {
					  dist[x_new][y_new]=dist[i][j]+1;
					  q.push({x_new,y_new});
				  }
			  }
		  }
		  return dist;   
	}
######  Mother Vertex 
	void dfs(int i,vector<int> g[],vector<int> &vis)
	 {
		if(vis[i]){return;}
		vis[i]=1;
		for(auto u:g[i])
		{
			if(!vis[u])
			{
				dfs(u,g,vis);
			}
		}
	 }
	int findMother(int v, vector<int> g[]) 
	{ 
	   int last_vis_node;
	   vector<int> vis(v,0);
		for(int i=0;i<v;i++)//dfs of all nodes;
		{   
			if(!vis[i])
			{
				dfs(i,g,vis);
				last_vis_node= i; 
			}
		}
		vis.clear();
		vis.resize(v);
		dfs(last_vis_node,g,vis);
		for(int i=0;i<v;i++)
		{
			if(!vis[i])
				return -1;
		}
		return (last_vis_node);
	} 
###### Unit Area of largest region of 1's
	void dfs(int a[][SIZE],int i,int j,int &curr,int n,int m)
	{
		if(i<0||j<0||i>=n||j>=m||a[i][j]!=1)
		return;
		curr++;
		a[i][j]=-1;
		dfs(a,i+1,j,curr,n,m);
		dfs(a,i-1,j,curr,n,m);
		dfs(a,i,j+1,curr,n,m);
		dfs(a,i,j-1,curr,n,m);
		dfs(a,i+1,j+1,curr,n,m);
		dfs(a,i+1,j-1,curr,n,m);
		dfs(a,i-1,j+1,curr,n,m);
		dfs(a,i-1,j-1,curr,n,m);

	}
	int findMaxArea(int n, int m, int a[SIZE][SIZE] )
	{
	 int ans=0,curr,i,j;
	 for(i=0;i<n;i++)
	 for(j=0;j<m;j++)
	 if(a[i][j]==1)
	 {
		 curr=0;
		 dfs(a,i,j,curr,n,m);
		 ans=max(ans,curr);
	 }
	 return ans;
	}
###### Rotten Oranges
	void rot(vector<vector<int>>&m, int r, int c, int x, int y, queue<pair<int, int>>&q, vector<vector<int>>&visited) {
		if(x+1<r && (m[x+1][y] == 1) && !visited[x+1][y]) {
			m[x+1][y] = 2;
			q.push({x+1, y});
		}
		if(x-1>=0 && m[x-1][y] == 1 && !visited[x-1][y]) {
			m[x-1][y] = 2;
			q.push({x-1, y});
		}
		if(y-1>=0 && m[x][y-1] == 1 && !visited[x][y-1]) {
			m[x][y-1] = 2;
			q.push({x, y-1});
		}
		if(y+1<c && m[x][y+1] == 1 && !visited[x][y+1]) {
			m[x][y+1] = 2;
			q.push({x, y+1});
		}
	}

	int rotOranges(vector<vector<int> > &m, int r, int c) {
		queue<pair<int, int>> q;
		vector<vector<int>> visited(r, vector<int>(c, 0));
		for(int i=0; i<r; i++) {
			for(int j=0; j<c; j++) {
				if(m[i][j] == 2)
					q.push({i, j});
			}
		}
		q.push({-1, -1});
		int ans = 0;
		while(!q.empty()) {
			pair<int, int> p = q.front();
			q.pop();
			int x = p.first, y = p.second;
			if(x < 0) {
				ans = max(ans, abs(x));
				if(!q.empty())
					q.push({x-1, y-1});
				continue;
			}
			visited[x][y] = 1;
			rot(m, r, c, x, y, q, visited);
		}
		for(int i=0; i<r; i++) {
			for(int j=0; j<c; j++) {
				// cout << m[i][j] << " ";
				if(m[i][j] == 1)
					return -1;
			}
			// cout << "\n";
		}
		return ans-1;
	}
###### Minimum Swaps to Sort
	int minSwaps(int a[], int n){
		int c=0,i,t;
		int temp[n];
		for(i=0;i<n;i++)
			temp[i]=a[i];
		sort(temp,temp+n);
		for(i=0;i<n;i++){
			a[i]=lower_bound(temp,temp+n,a[i])-temp;
		}
		for(i=0;i<n-1;i++){

			while(i!=a[i]){
				t=a[a[i]];
				a[a[i]]=a[i];
				a[i]=t;
				c+=1;

			}
		}
		return c;
	}
###### Steps by Knight
	knight(n,xi-1,yi-1);
	cout<<mat[xf-1][yf-1]<<endl;

	bool safe(int n,int x,int y){
		return x>=0 && y>=0 && x<n && y<n;
	}
	void knight(int n,int xi,int yi){
		int row[]={-2,-2,-1,-1,2,2,1,1};
		int col[]={-1,1,-2,2,-1,1,-2,2};
		int k;
		queue<pair<int,int>>Q;
		Q.push(make_pair(xi,yi));
		while(!Q.empty()){
			auto x=Q.front();
			Q.pop();
			visited[x.first][x.second]=1;
			for(k=0;k<8;k++){
				if(safe(n,x.first+row[k],x.second+col[k]) && visited[x.first+row[k]][x.second+col[k]]==0){
					mat[x.first+row[k]][x.second+col[k]]=mat[x.first][x.second]+1;
					Q.push(make_pair(x.first+row[k],x.second+col[k]));
				}
			}
		}
	}
###### Minimum Cost Path
	bfs();
	cout<<level[n-1][n-1]<<"\n";

	int is_valid(int x, int y)
	{
		if(x>=0 && y>=0 && x<n && y<n ) return 1;
		else return 0;   
	}
	void bfs()
	{
		for(int i=0;i<n;i++)
		{
			for(int j=0;j<n;j++)
			{
				vis[i][j]=0;
				level[i][j]=INT_MAX;
			}
		}
		queue<pair<int,int> > q;
		q.push({0,0});
		vis[0][0]=1;
		level[0][0]=arr[0][0];
		while(!q.empty())
		{
			pair<int,int> v;
			v=q.front();
			q.pop();
			for(int i=0;i<4;i++)
			{
				int nx = v.first + dx[i];
				int ny = v.second +dy[i];
				if(is_valid(nx,ny) && level[nx][ny]>level[v.first][v.second]+arr[nx][ny])
				{
					level[nx][ny]=level[v.first][v.second]+arr[nx][ny];
					vis[nx][ny]=1;
					q.push({nx,ny});
				}
			}
		}
	}
###### Implementing Dijkstra | Set 1 (Adjacency Matrix)

	void dijkstra(vector<vector<int>> graph, int src, int V)
	{
		// Your code here
		priority_queue < pair<int,int> , vector<pair<int,int>> , greater<pair<int,int>> > pq;
		vector<int>distance(V,INT_MAX);
		bool visited[V]={false};
		pq.push({0,src});
		distance[src]=0;

		while(!pq.empty()){
			pair<int,int> p=pq.top();
			pq.pop();
			int u=p.second;
			if(visited[u]==true) continue;
			visited[u]=true;
			for(int i=0;i<V;i++){
				if(graph[u][i]!=0){
					int v=i;    
					int wt=graph[u][i];
					if(distance[v]>distance[u]+wt){
						distance[v]=distance[u]+wt;
						pq.push({distance[v],v});
					}
				}
			} 
		}
		for(int i=0;i<V;i++) cout<<distance[i]<<" ";
	}
###### Minimum Spanning Tree

	int spanningTree(int  V,int E,vector<vector<int> > graph)
	{
		int vis[V],i,summ;
		memset(vis,0,sizeof(vis));
		priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
		pq.push(make_pair(0,0));
		summ=0;
		while(!pq.empty()){
			auto x=pq.top();
			pq.pop();
			if(vis[x.second])
				continue;
			summ+=x.first;
			vis[x.second]=1;
			for(i=0;i<V;i++){
				if(graph[x.second][i]!=INT_MAX && vis[i]==0){
					pq.push(make_pair(graph[x.second][i],i));
				}
			}
		}
		return summ;
	}
###### Strongly Connected Components (Kosaraju's Algo)
	void tdfs(vector<int> adj[], bool vis[], int u,stack<int> &s){
		vis[u] = 1;
		for(auto itr = adj[u].begin();itr!=adj[u].end();itr++){
			if(!vis[*itr])
				tdfs(adj,vis,*itr,s);
		}
		s.push(u);
	}
	void dfs(vector<int> adjt[], bool vis[], int start){
		vis[start] = 1;
		for(auto itr = adjt[start].begin();itr!=adjt[start].end();itr++){
			if(!vis[*itr])
				dfs(adjt,vis,*itr);
		}
	}
	int kosaraju(int V, vector<int> adj[])
	{
		int res = 0;
		bool vis[V] = {};
		stack<int> s;

		for(int i=0;i<V;i++){
			if(!vis[i]){
				tdfs(adj,vis,i,s);
			}
		}
		vector<int> adjt[V];
		for(int i=0;i<V;i++)
			vis[i] = 0;
		for(int u=0;u<V;u++){
			for(auto itr = adj[u].begin();itr!=adj[u].end();itr++){
				adjt[*itr].push_back(u);
			}
		}
		while(!s.empty()){
			int t = s.top();
			s.pop();
			if(!vis[t]){
				dfs(adjt,vis,t);
				res++;
			}
		}
		return res;
	}
###### Bridge Edge in Graph
	bool util(int u,list<int> adj[], int V, int s, int e,vector<int>&disc,vector<int>&low,vector<int>&par)
	{
	 static int tc=0;
	 disc[u]=low[u]=++tc;
	 for(auto it=adj[u].begin();it!=adj[u].end();it++)
	 {
		 int v=*it;
		 if(disc[v]==-1)
		 {
			 par[v]=u;
			 if(util(v,adj,V,s,e,disc,low,par))
			 return true;
			 low[u]=min(low[u],low[v]);
			 if(low[v]>low[u])
			 {
				 if(u==s&&v==e||v==s&&u==e) 
					 return true;
			 }
		 }
		 else if(par[u]!=v)
		 {
			 low[u]=min(low[u],disc[v]);
		 }
	 }
	 return false;
	}
	bool isBridge(list<int> adj[], int V, int s, int e) {
		vector<int> disc(V,-1);
		vector<int> low(V,-1);
		vector<int> par(V,-1);
		for(int i=0;i<V;i++)
		if(disc[i]==-1)
			if(util(i,adj,V,s,e,disc,low,par)) 
				return true;
		return false;
	}
###### Strongly connected component (Tarjans's Algo)
	stack<pair<int,int>> st;
	void dfs(int s, vector<int> adj[]){
		low[s] = id[s] = ++t;
		vis[s] = true;
		inStack[id[s]] = true;
		st.push(make_pair(id[s], s));

		for(int u : adj[s]){
			if(vis[u] == false){
				dfs(u, adj);
			}
			if(inStack[id[s]] && inStack[id[u]])
					low[s] = min(low[s], low[u]);
		}

		if(low[s] == id[s]){
			while(st.top().first != id[s]){
				cout<<st.top().second<<" ";
				inStack[st.top().first] = false;
				st.pop();
			}
			inStack[st.top().first] = false;
			cout<<st.top().second<<",";
			st.pop();
		}
	}
	void find(vector<int> adj[], int N) 
	{

		 t = -1;
		memset(vis, false, sizeof(vis));
		memset(inStack, false, sizeof(inStack));
		while(!st.empty())
			st.pop();
		for(int i=0;i<1000001; i++)
			low[i] = i;

		for(int i=0; i<N; i++)
			if(vis[i] == false)
				dfs(i, adj);
	}
## Greeddy

######  Activity Selection 
	bool myCmp(pair<int, int> a, pair<int, int> b)
		return a.second < b.second;
	int activitySelection(int start[], int end[], int n) {
		vector<pair<int, int>> jobTime;
		for(int i = 0; i < n; i++) {
			jobTime.push_back({start[i], end[i]});
		}
		sort(jobTime.begin(), jobTime.end(), myCmp);
		int count = 1;
		pair<int, int> last = jobTime[0];
		for(int i = 1; i < n; i++)
		{
			if(jobTime[i].first >= last.second)
			{
				count++;
				last = jobTime[i];
			}
		}
		return count;
	}

###### Huffman Decoding
	struct MinHeapNode
	{
		char data;
		int freq;
		MinHeapNode *left, *right;
	};

	typedef struct MinHeapNode Node;
	string decodeHuffmanData(struct MinHeapNode* root, string s)
	{
		string o;
		for(int i=0;i<s.length();)
		{
			Node *t=root;
			while(t->left&&t->right)
			{
				if(s[i]=='0')
				t=t->left;
				else
				t=t->right;
				i++;
			}
			o+=t->data;
		}
		return o;
	}
###### Fractional Knapsack
	bool cmp(struct Item a, struct Item b) 
	{ 
		double r1 = (double)a.value / a.weight; 
		double r2 = (double)b.value / b.weight; 
		return r1 > r2; 
	} 

	double fractionalKnapsack(int W, Item arr[], int n)
	{           
		sort(arr,arr+n,cmp);int cw=0;
		double profit=0.0;
		for(int i=0;i<n;i++)
		{   //if(W==0)break;
			if((cw+arr[i].weight)<=W)
			{   profit=profit+arr[i].value;
				cw=cw+arr[i].weight;
			}

			else
			{   int remain=W-cw;
				profit=profit+((double)remain/arr[i].weight)*arr[i].value;
				break;
			}
		}
		return profit;
	}
###### Largest number with given sum
	string largestNumber(int n, int sum){
		string larnum="";
		if(sum>(n*9))
		{    larnum=to_string(-1);return larnum;}
		for(int i=0;i<n;i++)
		{   if(sum==0)
			{   larnum=larnum+'0';}
			else if(sum>=9)
			{    larnum=larnum+'9';sum=sum-9;}
			else if(sum<9)
			{   larnum=larnum+to_string(sum);sum=0;}
		}
		return larnum;
	}
###### Job Sequencing Problem
	bool cmp(struct Job a,struct Job b)
	{   return a.profit>b.profit;
	}
	pair<int,int> JobScheduling(Job arr[], int n) 
	{       sort(arr,arr+n,cmp);
			pair<int,int>p;
			int slot[n];
			bool ji[n];
			for(int i=0;i<n;i++)
			{   ji[i]=false;}

			for(int i=0;i<n;i++)
			{   for(int j=min(n,arr[i].dead)-1;j>=0;j--)
				{   if(ji[j]==false)
					{   ji[j]=true;
						slot[j]=i;
						break;
					}
				}
			}
			int profit=0;
			int jo=0;
			for(int i=0;i<n;i++)
			{   if(ji[i])
				{   profit=profit+arr[slot[i]].profit;
					jo++;
				}
			}
			p.first=jo;
			p.second=profit;
			return p;
	} 

## Backtracking

######  Rat in a Maze Problem - I 
	vector<string> v;
	bool visited[100][100];
	string s;
	bool isvalid(int i,int j,int n)
	{
		if(i>=0&&i<n&&j>=0&&j<n)
		return true;
		return false;
	}
	void path(int m[MAX][MAX],int i,int j,int n)
	{
		visited[i][j]=true;
		if(i==n-1&&j==n-1)
		v.push_back(s);
		if(m[i+1][j]&&isvalid(i+1,j,n)&&!visited[i+1][j])
		{
			s=s+"D";
			path(m,i+1,j,n);
			s.pop_back();
			visited[i+1][j]=false;
		}
		if(m[i][j+1]&&isvalid(i,j+1,n)&&!visited[i][j+1])
		{
			s=s+"R";
			path(m,i,j+1,n);
			s.pop_back();
			visited[i][j+1]=false;
		}
		if(m[i-1][j]&&isvalid(i-1,j,n)&&!visited[i-1][j])
		{
			s=s+"U";
			path(m,i-1,j,n);
			s.pop_back();
			visited[i-1][j]=false;
		}
		if(m[i][j-1]&&isvalid(i,j-1,n)&&!visited[i][j-1])
		{
			s=s+"L";
			path(m,i,j-1,n);
			s.pop_back();
			visited[i][j-1]=false;
		}
	}
	vector<string> printPath(int m[MAX][MAX], int n)
	{
		v.clear();
		s.clear();
		for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
		visited[i][j]=false;
		path(m,0,0,n);
		sort(v.begin(),v.end());
		return v;	
	}
###### Rat Maze With Multiple Jumps
	bool isSafe(vector<int> maze[], int i, int j, int N)
	{
		return i < N && j < N && maze[i][j] != 0;
	}
	bool solRec(int i, int j, vector<int> maze[], vector<int> sol[], int N)
	{
		if(i == N-1 && j == N-1)
		{
			sol[i][j] = 1;
			return true;
		}
		if(isSafe(maze, i, j, N) == true)
		{
			int jumps = maze[i][j];
			sol[i][j] = 1;   
			for(int k = 1; k <= jumps; k++)
			{
				if(solRec(i, j+k, maze, sol, N) == true)
				{
					return true;
				}
				if(solRec(i+k, j, maze, sol, N) == true)
				{
					return true;
				}
			}       
			sol[i][j] = 0;
		}    
		return false;
	}

	void solve(int N, vector<int> maze[]) 
	{
		vector<int> sol[N];
		for(int i = 0; i < N; i++)
			sol[i].assign(N, 0);
		if(solRec(0, 0, maze, sol, N) == false)
			cout << "-1\n";
		else
			print(N, sol);
	}
###### Black and White
	long long solve(int n, int m) {
		int x_off[] = {-2, -2, -1, 1, 2, 2, 1, -1};
		int y_off[] = {-1, 1, 2, 2, 1, -1, -2, -2};
		  long long MOD = 1e9 + 7;
		long long ret = 0;
		int x, y;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				for (int k = 0; k < 8; ++k) {
					x = i + x_off[k];
					y = j + y_off[k];
					// checking if the attack position is within bounds
					if (x >= 0 && x < n && y >= 0 && y < m)
						++ret; // if in bounds it is not feasible, increment it
				}
			}
		}
		long long total = n * m;
		total =
			(total * (total - 1)) ; // total possible combinations of 2 knights
		return (total - ret) % MOD; // returning total feasible combinations
	}
###### 
###### 
###### 
###### 
###### 
