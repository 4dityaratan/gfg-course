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
