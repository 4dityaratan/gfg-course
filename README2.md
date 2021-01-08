# Data Structure (Advanced)
## Bitmagic

## Recursion
###### Josephus problem
    int josephus(int n, int k)
    {
       if(n==1)return 1;
       return (josephus(n-1,k)+k-1)%n+1;
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
    
            if(v[mid] <=x && (mid==n-1 || v[mid+1] > x))


###### Left Index
    
            if((arr[mid] == elementToSearch) && ((mid == 0) || arr[mid-1] != elementToSearch))
            {
                return mid;
            }
            
## Sorting

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

## Strings

###### Distinct Pattern Search
	bool search(string pat, string txt) 
	{ 
	    if(txt.find(pat) != string::npos)
		    return true;

	    return false;

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
###### Combination Sum
	void findNumbers(vector<int>& ar, int sum, vector<vector<int>>& res, vector<int>& r, int i) 
	{ 
	    if (sum < 0) 
		return; 
	    if (sum == 0) 
	    { 
		res.push_back(r); 
		return; 
	    } 
	    while (i < ar.size() && sum - ar[i] >= 0) 
	    { 
		r.push_back(ar[i]); // add them to list 
		findNumbers(ar, sum - ar[i], res, r, i); 
		i++; 
		r.pop_back(); 
	    } 
	} 
	vector<vector<int> > combinationSum(vector<int> &A, int B) {
	    sort(A.begin(),A.end());
	    A.erase(unique(A.begin(), A.end()), A.end()); 
	    vector<int> r; 
	    vector<vector<int> > res; 
	    findNumbers(A, B, res, r, 0); 
	    return res; 
	}

###### Subsets
	void findSub(vector<vector<int>> &res,vector<int> &subset,vector <int> &A,int index ){
	    res.push_back(subset);
	    for(int i=index;i<A.size();i++){
		if(i!=index && A[i]==A[i-1])
		    continue;
		subset.push_back(A[i]);
		findSub(res,subset,A,i+1);
		subset.pop_back();
	    }
	}
	void func (vector <int> A)
	{
	    vector<vector<int>> res;
	    vector<int> subset;
	    sort(A.begin(),A.end());
	    findSub(res,subset,A,0);
	     for(auto x:res){
		cout<<"(";
		for(int i = 0; i<x.size(); i++)
		    if(i==x.size()-1)cout<<x[i];
		    else cout<<x[i]<<" ";
		cout<<")";
	    }
	    cout<<endl;    
	}
###### M-Coloring Problem
	bool isSafe(bool graph[101][101],int i,int j,int V,vector<int> &color){
	    for(int k=0;k<V;k++){
		if(graph[i][k]==1 && color[k]==j) return false;

	    }
	    return true;
	}
	bool checkColor(bool graph[101][101], int m, int V,vector<int> &color,int i){
	    if(i==V) return true;
	    for(int j=0;j<m;j++){
		if(isSafe(graph,i,j,V,color)){
		    color[i]=j;
		    if(checkColor(graph,m,V,color,i+1))
			return true;
		    color[i]=-1;
		}
	    }
	    return false;
	}
	bool graphColoring(bool graph[101][101], int m, int V) {
	    vector<int> color(V,-1);
	    return checkColor(graph,m,V,color,0);
	}
###### Solve the Sudoku
	bool isSafe(int grid[N][N],int i,int j,int n){
	    for(int k=0;k<N;k++){
		if(grid[i][k]==n || grid[k][j]==n)
		    return false;
	    }
	    int sub=sqrt(N);
	    int rsub=i-i%sub;
	    int lsub=j-j%sub;
	    for(int k=0;k<sub;k++){
		for(int l=0;l<sub;l++){
		    if(grid[k+rsub][l+lsub]==n)
			return false;
		}
	    }
	    return true;
	}
	bool SolveSudoku(int grid[N][N])  
	{ 
	    int i,j;
	    for(i=0;i<N;i++){
	       bool flag=false;
		for(j=0;j<N;j++){
		    if(grid[i][j]==0){
			flag=true;
			break;
		    }
		}
		if(flag)
		    break;
	    }
	    if(i==N && j==N)
		return true;
	    for(int k=1;k<=9;k++){
		if(isSafe(grid,i,j,k)){
		grid[i][j]=k;
		if(SolveSudoku(grid))
		    return true;
		grid[i][j]=0;
		}
	    }
	    return false;
	}
	void printGrid (int grid[N][N]) 
	{
	    for(int i=0;i<N;i++){
		for(int j=0;j<N;j++)
		    cout<<grid[i][j]<<" ";
	    }
	}

## Dynamic Programming
###### Coin Change - Minimum number of coins
	
	    for (int i=1; i<=V; i++) 
		      for (int j=0; j<m; j++) 
              if (coins[j] <= i) 
              { 
                  int sub_res = dp[i-coins[j]]; 
                  if (sub_res != INT_MAX && sub_res + 1 < dp[i]) 
                  dp[i] = sub_res + 1; 
              }   
###### Coin Change - Number of ways

	    for(auto coin:coinsSet) //Using a coin, one at a time
          for(int i=1;i<value+1;i++)
              if(i>=coin) //Since it makes no sense to create change for value smaller than coin's denomination
                 ways[i]=ways[i]+ways[i-coin];

###### nCr
	int nCrModp(int n, int r) 
	{ 
		if(n<r) return 0;
		if(r==0) return 1;
		if(n==r) return 1;
		if(r>n-r) r=n-r;
		long long dp[r+1]={0};
	  dp[0]=1;
	  for(int i=1;i<=n;i++)
	  {
		  for(int j=min(i,r);j>0;j--)
		  dp[j] = (dp[j]%1000000007+dp[j-1]%1000000007)%1000000007;
	  }
	  return dp[r]; 
	} 

###### Unique BST's
	int numTrees(int n) {
		if(n==0||n==1)
			return 1;    
		int dp[n];
		dp[0]=dp[1]=1;
		for(int i=2;i<=n;i++)
		{
			dp[i]=0;
			for(int k=0;k<i;k++)
				dp[i]+=dp[k]*dp[i-k-1];
		}    
		return dp[n];
	}
###### Sum of all substrings of a number
	long long sumSubstrings(string s)
	{   
		long long n=s.size();
		long long prev_res[n];
		prev_res[0]=s[0]-'0'; 
		long long res =prev_res[0];
		for(int i = 1; i < s.size(); i++)
		{
			prev_res[i]= (((i+1)*(s[i] - '0') + 10 * prev_res[i-1]))%1000000007;
			res = (res + prev_res[i])%1000000007;
		}    
		return res;
	}
###### Max sum subarray by removing at most one element
	int maxSumSubarray(int a[], int n)
	{
		int sum=0,ans=INT_MIN;
		for(int i=0;i<n;i++)ans=max(ans,a[i]);
		for(int i=0;i<n;i++){
			sum+=a[i];
			ans=max(ans,sum);
			if(sum<0)sum=0;
		}
		for(int i=0;i<n;i++){
			if(a[i]<0){
				int sum=0;
				for(int j=0;j<n;j++){
					if(j==i)continue;
					sum+=a[j];
					ans=max(ans,sum);
					if(sum<0)sum=0;
				}
			}
		}
		return ans;
	}
###### Shortest Common Supersequence
	int scs(string str1, string str2, int l1, int l2,int dp[100][100]) {
		if(l1 == 0 ) return (l2);
		if(l2 == 0) return l1;
		if(dp[l1-1][l2-1] != 0)
			return dp[l1-1][l2-1];
		int len;
		if(str1[l1-1] == str2[l2-1]) {
			len = 1 + scs(str1, str2, l1-1, l2-1,dp);
		}
		else {
			len = min(scs(str1,str2,l1-1,l2,dp)+1, scs(str1,str2,l1,l2-1,dp)+1);
		}
		dp[l1-1][l2-1] = len;
		return len;

	}
###### Subset Sum Problem
	bool SUBSET(int n, int sum, int A[])
	{
		bool sub[n+1][sum+1];
		for(int i = 0;i<=n;i++)
			sub[i][0]=true;
		for(int j = 1;j<=sum;j++)
			sub[0][j]=false;
		for(int i = 1;i<=n;i++)
		{
			for(int j = 1;j<=sum;j++)
			{
				if(A[i-1]<=j)
					sub[i][j]=sub[i-1][j-A[i-1]] || sub[i-1][j]; 
				else
					sub[i][j]=sub[i-1][j];
			}
		}
		return sub[n][sum];
	}
###### Maximize The Cut Segments
	int maximizeTheCuts(int n, int x, int y, int z)
	{
		int dp[n+1];
		dp[0]=0;
		for(int i=1;i<=n;i++) dp[i] =-1;
		for(int i=1;i<=n;i++)
		{
			if(i-x>=0) dp[i] = max(dp[i],dp[i-x]);
			if(i-y>=0) dp[i] =max(dp[i],dp[i-y]);
			if(i-z>=0) dp[i] =max(dp[i],dp[i-z]);
			if(dp[i]!=-1) dp[i]++;
		}
		return (dp[n]>0)? dp[n]:0;
	}

###### Optimal Strategy For A Game
	long long maximumAmount(int arr[], int n) 
	{
	   long long int dp[n][n];
		dp[n-1][n-1]=arr[n-1];
		for(int i=0;i<n-1;i++)
		{  
			dp[i][i+1] = max(arr[i],arr[i+1]);
			dp[i][i]=arr[i];
		}
		for(int gap=3;gap<n;gap=gap+2)
		{
			for(int i=0;i+gap<n;i++)
			{
			  int j=i+gap;
				dp[i][j]= max(arr[i]+min(dp[i+2][j],dp[i+1][j-1]),arr[j]+min(dp[i+1][j-1],dp[i][j-2]));
			}
		}
		return dp[0][n-1];
	}
###### Egg Dropping Puzzle
	int eggDrop(int n, int k) 
	{
		int dp[n+1][k+1];
		for(int i=0; i<=n; i++)
		{
			dp[i][0] = 0;
			dp[i][1] = 1;
		}   
		for(int i=0; i<=k; i++)
			dp[1][i] = i;
		for(int i=2; i<=n; i++)
			for(int j=2; j<=k; j++)
			{
				dp[i][j] = INT_MAX;
				for(int x=1; x<=j; x++)
					dp[i][j] = min(dp[i][j],max(dp[i][j-x],dp[i-1][x-1]));
				dp[i][j]++;
			}
		return dp[n][k];
	}
## Segment Trees

######  Range Sum Queries 
	int construct(int ss, int se,int si,int arr[],int tree[]){
		if(ss == se)
		{
			tree[si] = arr[ss];
			return tree[si];
		}
		int mid = (ss+se)/2;
		tree[si] = construct(ss,mid,2*si+1,arr,tree) + construct(mid+1,se,2*si+2,arr,tree);
		return tree[si];   
	}
	int sum(int ss, int se,int si, int qs,int qe,int tree[]){
		if(se < qs || ss > qe)
			return 0;
		if(ss >= qs && se <= qe)
			return tree[si];
		int mid = (ss+se)/2;
		return sum(ss,mid,2*si+1,qs,qe,tree) + 
				sum(mid+1,se,2*si+2,qs,qe,tree);   
	}
	void update(int ss,int se,int si,int pos,int mf, int tree[]){ 
		if(ss > pos || se < pos)
			return;
		if(ss == se)
		{
			tree[si] += mf;
			return;
		}    
		tree[si] += mf;
		int mid = (ss+se)/2;
		update(ss,mid,2*si+1,pos,mf,tree);
		update(mid+1,se,2*si+2,pos,mf,tree);

	}

###### Range Min Max Queries
	pair<int,int> make_tree(int ss,int se,int si,int arr[],pair<int,int> tree[]){
		if(ss == se)
		{
			tree[si] = {arr[ss],arr[ss]};
			return tree[si];
		}
		int mid = ss  + (se-ss)/2;
		pair<int,int> l = make_tree(ss,mid,2*si+1,arr,tree);
		pair<int,int> r = make_tree(mid+1,se,2*si+2,arr,tree);
		tree[si] = {min(l.first,r.first),max(l.second,r.second)};
		return tree[si];   
	}
	pair<int,int> range(int ss,int se,int si,int qs,int qe,pair<int,int> tree[]){
		if(ss > qe || se < qs)
			return {INT_MAX,0};
		if(ss >= qs && se <= qe)
			return tree[si];
		int mid = ss  + (se-ss)/2;
		pair<int,int> l = range(ss,mid,2*si+1,qs,qe,tree);
		pair<int,int> r = range(mid+1,se,2*si+2,qs,qe,tree);
		return {min(l.first,r.first),max(l.second,r.second)};   
	}
	pair<int,int> update(int ss,int se, int si,int pos,int mf,pair<int,int> tree[]){
		if(ss > pos || se < pos)
			return tree[si];
		if(ss == se)
		{
			tree[si] = {mf,mf};
			return tree[si];
		}
		int mid = ss  + (se-ss)/2;
		pair<int,int> l = update(ss,mid,2*si+1,pos,mf,tree);
		pair<int,int> r = update(mid+1,se,2*si+2,pos,mf,tree);
		tree[si] = {min(l.first,r.first),max(l.second,r.second)};
		return tree[si];

	}
###### Range Longest Correct Bracket Subsequence Queries
	cin>>l>>r;
	int j=0;
	for(int i=l;i<=r;i++){
		if(s[i]=='('){
		mystack.push('(');
		myvector.push_back(j);
		}
		else if(s[i]==')' && !mystack.empty()){
			mystack.pop();
			j=j+2;
			myvector.push_back(j);
		}
		else if(s[i]==')' && mystack.empty()){
			myvector.push_back(j);
		}

	}
	cout<<*(myvector.end()-1)<<endl;
###### Range GCD Queries
	int make_tree(int ss,int se,int si,int arr[],int tree[]){
		if(ss == se){
			tree[si] = arr[ss];
			return tree[si];
		}
		int mid = ss + (se-ss)/2;
		tree[si] = __gcd(make_tree(ss,mid,2*si+1,arr,tree), make_tree(mid+1,se,2*si+2,arr,tree));
		return tree[si];
	}

	int get_gcd(int ss,int se,int si,int qs,int qe,int tree[]){
		if(ss > qe || se < qs){
			return -1;
		}
		if(ss >= qs && se <= qe)
			return tree[si];
		int mid = ss + (se-ss)/2;
		int le = get_gcd(ss,mid,2*si+1,qs,qe,tree);
		int ri = get_gcd(mid+1,se,2*si+2,qs,qe,tree);
		if(le == -1 || ri == -1)
			return max(le,ri);
		else
			return __gcd(le,ri);
	}

	int update(int ss,int se,int si,int pos,int mf,int tree[]){
		if(ss > pos || se < pos)
			return tree[si];
		if(ss == se){
			tree[si] = mf;
			return tree[si];
		}
		int mid = ss + (se-ss)/2;
		int le = update(ss,mid,2*si+1,pos,mf,tree);
		int ri = update(mid+1,se,2*si+2,pos,mf,tree);
		tree[si] = __gcd(le,ri);
		return tree[si];
	}
###### Largest Sum Contiguous Subarray in Range
	// Initial Template for C++

	#include <bits/stdc++.h>

	using namespace std;

	// Structure of node of the tree
	struct node {
		int sum, prefixSum, suffixSum, maxSubArraySum;

		node() { sum = prefixSum = suffixSum = maxSubArraySum = INT_MIN; }
	};

	// Utility function to build the tree
	void build(int arr[], node tree[], int low, int high, int index) {
		if (low == high) {
			tree[index].sum = arr[low];
			tree[index].prefixSum = arr[low];
			tree[index].suffixSum = arr[low];
			tree[index].maxSubArraySum = arr[low];
		} else {
			int mid = (low + high) / 2;
			build(arr, tree, low, mid, 2 * index + 1);
			build(arr, tree, mid + 1, high, 2 * index + 2);
			tree[index].sum = tree[2 * index + 1].sum + tree[2 * index + 2].sum;
			tree[index].prefixSum =
				max(tree[2 * index + 1].prefixSum,
					tree[2 * index + 1].sum + tree[2 * index + 2].prefixSum);
			tree[index].suffixSum =
				max(tree[2 * index + 2].suffixSum,
					tree[2 * index + 2].sum + tree[2 * index + 1].suffixSum);
			tree[index].maxSubArraySum = max(
				{tree[index].prefixSum, tree[index].suffixSum,
				 tree[2 * index + 1].maxSubArraySum,
				 tree[2 * index + 2].maxSubArraySum,
				 tree[2 * index + 1].suffixSum + tree[2 * index + 2].prefixSum});
		}
	}

	// function should update the array and as Tree as well accordingly
	void update(int arr[], node tree[], int n, int index, int new_value);

	// function should return the Max-Sum in the range
	int query(int arr[], node tree[], int n, int l, int r);

	// Driver Code
	int main() {
		int T;
		cin >> T;
		while (T--) {
			int n, q, index, value, left, right, type;
			int *arr = NULL;
			cin >> n >> q;
			arr = new int[n];
			node tree[n * 4];
			for (int i = 0; i < n; i++) cin >> arr[i];
			build(arr, tree, 0, n - 1, 0);
			for (int i = 0; i < q; i++) {
				cin >> type;
				if (type == 1) {
					cin >> left >> right;
					cout << query(arr, tree, n, left, right) << endl;
				} else {
					cin >> index >> value;
					update(arr, tree, n, index, value);
				}
			}
			delete[] arr;
			arr = NULL;
		}
		return 0;
	}// } Driver Code Ends


	// User funciton template in C++

	/*
	struct node {
		int sum, prefixSum, suffixSum, maxSubArraySum;

		node() {
			sum = prefixSum = suffixSum = maxSubArraySum = INT_MIN;
		}
	};
	*/

	// arr: given array
	// tree: segment tree
	// n: size of the array
	// update index value in arr to new_value
	void q(node tree[],int sb, int se, int l, int r,int si,node &x){
		if(l>se||r<sb)
		 return;
		if(sb>=l&&se<=r)
		{
			if(x.sum==INT_MIN){
				x=tree[si];
				return;
			}
			node y;
			y.sum=x.sum+tree[si].sum;
			y.prefixSum=max(x.prefixSum,x.sum+tree[si].prefixSum);
			y.suffixSum=max(tree[si].suffixSum,tree[si].sum+x.suffixSum);
			y.maxSubArraySum=max(y.prefixSum,y.suffixSum);
			y.maxSubArraySum=max(y.maxSubArraySum,x.maxSubArraySum);y.maxSubArraySum=max(y.maxSubArraySum,tree[si].maxSubArraySum);
			y.maxSubArraySum=max(y.maxSubArraySum,x.suffixSum+tree[si].prefixSum);
			x=y;
			return;
		}
		int mid=sb+(se-sb)/2;
		q(tree,sb,mid,l,r,2*si+1,x);
		q(tree,mid+1,se,l,r,2*si+2,x);
	}
	void up(node tree[],int sb,int se,int si,int index,int new_value){
		if(sb==se&&sb==index){
			tree[si].sum=new_value;
			tree[si].prefixSum=new_value;
			tree[si].suffixSum=new_value;
			tree[si].maxSubArraySum=new_value;
			return;
		}
		int mid=sb+(se-sb)/2;
		if(index>=sb&&index<=mid){
			up(tree,sb,mid,2*si+1,index,new_value);
			node x=tree[2*si+1];
			node y=tree[2*si+2];
			tree[si].sum=x.sum+y.sum;
			tree[si].prefixSum=max(x.prefixSum,x.sum+y.prefixSum);
			tree[si].suffixSum=max(y.suffixSum,y.sum+x.suffixSum);
			tree[si].maxSubArraySum=max(tree[si].prefixSum,tree[si].suffixSum);
			tree[si].maxSubArraySum=max(tree[si].maxSubArraySum,x.maxSubArraySum);tree[si].maxSubArraySum=max(y.maxSubArraySum,tree[si].maxSubArraySum);
			tree[si].maxSubArraySum=max(tree[si].maxSubArraySum,x.suffixSum+y.prefixSum);
		}
		else if(index>=mid+1&&index<=se){
			up(tree,mid+1,se,2*si+2,index,new_value);
			node x=tree[2*si+1];
			node y=tree[2*si+2];
			tree[si].sum=x.sum+y.sum;
			tree[si].prefixSum=max(x.prefixSum,x.sum+y.prefixSum);
			tree[si].suffixSum=max(y.suffixSum,y.sum+x.suffixSum);
			tree[si].maxSubArraySum=max(tree[si].prefixSum,tree[si].suffixSum);
			tree[si].maxSubArraySum=max(tree[si].maxSubArraySum,x.maxSubArraySum);tree[si].maxSubArraySum=max(y.maxSubArraySum,tree[si].maxSubArraySum);
			tree[si].maxSubArraySum=max(tree[si].maxSubArraySum,x.suffixSum+y.prefixSum);
		}
		return;
	}
	void update(int arr[], node tree[], int n, int index, int new_value) {
		// code here
		arr[index-1]=new_value;
		up(tree,0,n-1,0,index-1,new_value);
	}

	// l and r are the range given in the problem
	int query(int arr[], node tree[], int n, int l, int r) {
		// code here
		node x;
		q(tree,0,n-1,l-1,r-1,0,x);
		return x.maxSubArraySum;
	}
###### Range LCM Queries
	int gcd(int a, int b){
		if(!b) return a;
		return gcd(b, a%b);
	}

	int lcm(int a, int b){
		return (a*b)/(a > b ? gcd(a,b) : gcd(b,a));
	}

	void buildTree(int l, int h, int idx){
		if(l > h) return;
		if(l == h) {
			tree[idx] = arr[l];
			return;
		}

		int mid = (l+h)>>1, lc = idx<<1, rc = lc|1;
		buildTree(l, mid, lc);
		buildTree(mid+1, h, rc);
		tree[idx] = lcm(tree[lc], tree[rc]);
	}

	void updateTree(int l, int h, int pos, int val, int idx){
		if(l > h || l > pos || h < pos) return;
		if(l == h && l == pos){
			tree[idx] = val;
			return;
		}

		int mid = (l+h)>>1, lc = idx<<1, rc = lc|1;
		updateTree(l, mid, pos, val, lc);
		updateTree(mid+1, h, pos, val, rc);
		tree[idx] = lcm(tree[lc], tree[rc]);
	}

	int query(int l, int h, int ql, int qh, int idx){
		if(l > h || l > qh || h < ql) return 0;
		if(l >= ql && h <= qh){ 
			return tree[idx];
		}

		int mid = (l+h)>>1, lc = idx<<1, rc = lc|1;
		int a = query(l, mid, ql, qh, lc);
		int b = query(mid+1, h, ql, qh, rc);
		if (!a) return b;
		if (!b) return a;
		return lcm(a, b);
	}
######  Union-Find 
	int findRoot(int i, int par[], int rank1[]) {
		while(i!=par[i]) {
			i = par[i];
		}
		return i;
	}
	void union_( int a, int b, int par[], int rank1[])
	{ 
		int x = findRoot(a, par, rank1);
		int y = findRoot(b, par, rank1);
		if(x==y)
		{
			return;
		}
		if(rank1[x]>=rank1[y])
		{
			rank1[x]++;
			par[y] = par[x];
		}
		else 
		{
			rank1[y]++;
			par[x] = par[y];
		}
		return;
	}
	bool isConnected(int x,int y, int par[], int rank1[]) 
	{ 
		return (findRoot(x, par, rank1) == findRoot(y, par, rank1));
	}
###### Number of Connected Components
	int find_root(int x,int par[])
	{
		if(x == par[x])
			return x;
		par[x] = find_root(par[x],par);
		return par[x];
	}
	void unionNodes( int a, int b, int  par[], int rank[], int n) {
		int x_rep = find_root(a,par);
		int y_rep = find_root(b,par);
		if(x_rep == y_rep)
			return;
		else if(rank[x_rep] < rank[y_rep])
			par[x_rep] = y_rep;
		else if(rank[x_rep] > rank[y_rep])
			par[y_rep] = x_rep;
		else
		{
			par[y_rep] = x_rep;
			rank[x_rep]++;
		}
	}
	int findNumberOfConnectedNodes( int n, int par[], int rank1[]) {   
		int count = 0;
		for(int i=1; i<=n; i++)
			if(i == par[i])
				count++;
		return count;
	}

###### Detect Cycle using DSU
	int find(int x,int parent[])
	{
		if(parent[x]==x)
		return x;
		parent[x]=find(parent[x],parent);
		return parent[x];
	}
	bool union_( int a, int b, int par[], int rank1[]) 
	{
		int a_rep=find(a,par);
		int b_rep=find(b,par);
		if(a_rep==b_rep)
		return true;
		if(rank1[a_rep]<rank1[b_rep])
		par[a_rep]=b_rep;
		else if(rank1[b_rep]<rank1[a_rep])
		par[b_rep]=a_rep;
		else
		{
			par[b_rep]=a_rep;
			rank1[a_rep]++;
		}
		return false;
	}
	bool findCycle(vector<int> adj[], int parent[], int rank1[], int n, int e) 
	{
		for(int i=1;i<n;i++)
		{
			for(auto it=adj[i].begin();it!=adj[i].end();it++)
			{
				if(i<*it && union_(i,*it,parent,rank1))
					return true;
			}
		}
		return false;
	}
###### Minimum Spanning Tree using Kruskal
	bool union1(int u,int v,int parent[],int rank1[])
	{
		while(parent[u]!=u)
			u=parent[u];
		while(parent[v]!=v)
			v=parent[v];
		if(u==v)
			return false;
		if(rank1[u]>rank1[v])
		{
			parent[v]=u;
			rank1[u]=rank1[u]+rank1[v];
		}
		else
		{
			parent[v]=u;
			rank1[v]=rank1[v]+rank1[u];
		}   
		return true;
	}
	long long int kruskalDSU(vector<pair<int, long long int>> adj[], int n, int m) {
		int visited[n+1]={0};
		vector<pair<long long int,pair<int,int>>>v;
		for(int i=1;i<=n;i++)
		{
			for(int j=0;j<adj[i].size();j++)
			{
				int u=i,v1=adj[i][j].first;
				long long int w=adj[i][j].second;
				if(visited[v1]==0)
				{
					v.push_back({w,{u,v1}});        
				}
			}
			visited[i]=1;
		}
		sort(v.begin(),v.end());
		long long int ans=0;
		int parent[n+1],rank1[n+1];
		for(int i=1;i<=n;i++)
		{
			parent[i]=i;rank1[i]=1;
		}
		n--;
		for(int i=0;i<v.size()&& n>0;i++)
		{
			int u,v1;
			long long int w;
			w=v[i].first;
			u=v[i].second.first;
			v1=v[i].second.second;
			if(union1(u,v1,parent,rank1))
			{
				ans=ans+w;n--;
			}   
		} 
		   return ans;
	}
