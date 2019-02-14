import os, time
import numpy as np
import h5py, re
import hashlib, fcntl
import random

'''
	 max_size_str - Digital+[K|M|G|T]
	 
	 h5py data 文件名: node_index.hdf5
		例:	12345555656544444_1.hdf5
		index 表示 node 文件的后续数据文件
		
		
	API:
		read()
		write()
		rewrite()
		add_one_node()
		belong_to_hash()
		merge()
'''

class wdh_log:

	hashring_max = 2**128
	
	def __init__( self, file_path, max_size_str='2g' ):
		self.max_size = self.__parse_max_size( max_size_str )
		
		self.file_path, self.node_list = file_path, np.array( [] )
		# init self.node_list - np array type
		self.__gather_nodes( self.file_path )

	
	def __parse_max_size( self, max_size_str ):
		pattern = '(\d+\.?\d*)\s*([KMGTkmgt])'
		res = re.match( pattern, max_size_str )
		if res:
			size = float( res[1] )
			unit = res[2].upper()
			if unit=='K':
				res = size * 1024
			elif unit=='M':
				res = size * 1024**2
			elif unit=='G':
				res = size * 1024**3
		else:
			print( 'invalid max_size_str. Set max_size to 2G' )
			res = 2 * 1024**3
			
		return int( res )
	
	
	def __gather_nodes( self, file_path ):
		node_list = []
		pattern = '(\d+)_?'
		for f in os.listdir( file_path ):
			res = re.match( pattern, f )
			if res:
				node_list.append( int(res[1]) )
				
		node_list = list( set(node_list) )
		node_list.sort()
		self.node_list = np.array( node_list )
	

	# 返回 value 归属的 node
	# node 值由 md5-128bits 计算
	def belong_to_hash( self, value ):
		if self.node_list.shape[0]<=0:
			return -1
	
		hash_md5 = hashlib.md5( value.encode('utf-8') )
		h_value = int( hash_md5.hexdigest(), 16 )
		
		res = np.where( self.node_list>=h_value )
		if res[0].shape[0]==0:
			res = self.node_list[0]
		else:
			res = self.node_list[ res[0][0] ]
		
		return res
	

	# 新增一个节点后, 只会波及一个 node
	# 返回受波及的 node, 此 node 即为 数据迁移时的 source node
	def __source_node( self, node ):
		res = np.where( self.node_list>node )
		if res[0].shape[0]==0:
			res = self.node_list[0]
		else:
			res = self.node_list[ res[0][0] ]
		return res
		
	
	def __open( self, file_name, mod='r+', wait=2 ):
		fid = open( os.path.join(self.file_path, file_name), mod )
		sleep_t = 0
		while sleep_t<wait:
			try:
				fcntl.flock( fid.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB )
				return fid
			except:
				time.sleep( 0.2 )
				sleep_t += 0.2
		return None
	
	
	# 从 ××_index.hdf5 读取数据
	def read( self, file_index, group, key ):
		belong_to_hash = self.belong_to_hash( group )
		the_file_name = str( belong_to_hash ) + '_' + str( file_index ) + '.hdf5'
		
		res = None
		fid = self.__open( the_file_name, mod='rb+' )
		if fid:
			f = h5py.File( fid )
			res = f[group+'/'+key][:]
			f.close()
			fid.close()
		return res
		
		
	# 向 ××_index.hdf5 写入数据
	# 如果 要写入的文件大小超出 max_size, 则自动新建新文件( 文件名中的index值自增加1 )
	def write( self, group, key, value_str_list ):
		belong_to_hash = self.belong_to_hash( group )
		new_file_name, the_file_name = self.__node_new_file( belong_to_hash )

		size = os.path.getsize( os.path.join(self.file_path,the_file_name) )
		if size>=self.max_size:
			os.mknod( os.path.join(self.file_path, new_file_name) ) 
			the_file_name = new_file_name
		
		res = False
		fid = self.__open( the_file_name, mod='rb+' )
		if fid:
			f = h5py.File( fid )
			
			g = f.require_group( group )
			dt = h5py.special_dtype( vlen=str )
			if key not in g.keys():
				g.create_dataset( key, (0,), dtype=dt, maxshape=(None, ) )
			
			value_str_array = np.array( value_str_list )
			uname = group+'/'+key
			end_index = f[uname].shape[0]
			f[uname].resize( (end_index+value_str_array.shape[0],) )
			f[uname][end_index:] = value_str_array
			
			f.close()
			fid.close()
			res = True
			
		return res

		
	# num - 增加的节点的数量
	# 新 node 为 node_list 中间距最大段的中点
	# 更新node_list, 自动建立文件, 并且自动转移数据
	def add_one_node( self ):
		if len( self.node_list )<=0:
			new_node = int( wdh_log.hashring_max-1 )
		else:
			res = []
			loop = self.node_list.shape[0]
			for i in range( 1, loop ):
				res.append( self.node_list[i]-self.node_list[i-1] )
			res.append( wdh_log.hashring_max+self.node_list[0]-self.node_list[-1] )
			
			index = np.where( np.array(res)==max(res) )
			index = index[0][0]
			if index==loop-1:
				new_node = int( (wdh_log.hashring_max+self.node_list[0]+self.node_list[-1])/2 ) % wdh_log.hashring_max
			else:
				new_node = int( (self.node_list[index-1]+self.node_list[index])/2 )
				
		self.node_list = np.append( self.node_list, new_node )
		self.node_list = np.sort( self.node_list )
		
		# 新建文件
		self.create_new_file( new_node )
		self.__migrate_data( new_node )
	

	# 将相关数据迁移到新节点文件中
	# 首先迁移到 new_node_1.hdf5 文件中, 如果超出最大文件限制, 自动递增新数据文件
	def __migrate_data( self, new_node ):
		source_node = self.__source_node( new_node )
		source_node_files = self.__node_all_file( source_node )
		
		for f_name in source_node_files:
			fid = self.__open( f_name, mod='rb+' )
			if fid:
				f = h5py.File( fid )
				for g in f.keys():
					belong_to_hash = self.belong_to_hash( g )
					if belong_to_hash==new_node:
						# 复制内容至当前新节点文件
						for k in f[g].keys():
							uname = g + '/' + k
							self.write( g, k, f[uname] )
						del f[g]
				f.close()
				fid.close()
				
				# 重写 source_node_files 以缩小文件
				self.rewrite( f_name )
			
		self.merge( source_node )


	def create_new_file( self, node ):
		new_file_name, _ = self.__node_new_file( node )
		os.mknod( os.path.join(self.file_path, new_file_name) )
		self.__gather_nodes( self.file_path )
		
	
	# 重新写入 file_name 文件,以达到缩小文件的目的
	# file_name - node_index.hdf5
	def rewrite( self, file_name ):
		src = os.path.join( self.file_path, file_name )
		
		mid = os.path.splitext( file_name )[0]
		temp = os.path.join( self.file_path, mid+'_t.hdf5' )
		os.rename( src, temp )
		os.mknod( src )
		
		f1 = h5py.File( temp, 'r' )
		f2 = h5py.File( src )
		
		for g in f1.keys():
			f2.copy( f1[g], f2 )
		f2.close()
		f1.close()
		os.remove( temp )
	
	
	# 将 node 文件尽量合并
	def merge( self, node ):
		node_files = self.__node_all_file( node )
		if len( node_files )<=1:
			return True	
		
		# 找出第一个可以接收新数据的文件
		src_files = node_files[:]
		for f in node_files:
			size = os.path.getsize( os.path.join(self.file_path,f) )
			if size<self.max_size:
				src_files.remove( f )
				break
			else:
				src_files.remove( f )
				
		# 剩余数据文件更名
		temp_list = []
		for src_f in src_files:
			src = os.path.join( self.file_path, src_f )
			
			mid = os.path.splitext( src_f )[0]
			temp = os.path.join( self.file_path, mid+'_t.hdf5' )
			temp_list.append( temp )
			os.rename( src, temp )
		
		for tf in temp_list:
			f = h5py.File( tf, 'r' )
			for group in f.keys():
				for k in f[group].keys():
					self.write( group, k, f[group][k][:] )
			f.close()
			os.remove( tf )
			
		return True
				
	
	# 返回 给定node 下一个index，和当前最大index 的文件名
	# 例:	如当前 node 存在文件 node_1, node_2, node_3
	#		则该函数返回 node_4, node_3
	def __node_new_file( self, node ):
		cur_index = 0
		pattern = str( node ) + '_?(\d*)\.'
		
		for f in os.listdir( self.file_path ):
			res = re.match( pattern, f )
			if res and res[1]!='':
				mid = int( res[1] )
				if mid>cur_index:
					cur_index = mid
			
		pre_node_file = str( node ) + '_' + str( cur_index ) + '.hdf5'
		node_file = str( node ) + '_' + str( cur_index+1 ) + '.hdf5'
		return node_file, pre_node_file
				
	
	# 返回按 index 升序的 文件名
	def __node_all_file( self, node ):
		index_list = []
		pattern = str( node ) + '_?(\d*)\.'
		
		for f in os.listdir( self.file_path ):
			res = re.match( pattern, f )
			if res and res[1]!='':
				index_list.append( int(res[1]) )
				
		index_list.sort()
		file_names = []
		for i in index_list:
			file_names.append( str(node)+'_'+str(i)+'.hdf5' )
			
		return file_names
			
		
	def __del__( self ):
		pass
		
		
if __name__=='__main__':
	
	'''
	wdh.write( 'user_100', 'scan-1', ['wangdehui====1', '123'] )
	d = wdh.read( 1, 'user_100', 'scan-1' )
	print( d )
	'''
	
	# 测试数据转移
	file_path = 'h5py_data'

	try:
		os.remove( file_path + '/' + str(old_node)+'_1.hdf5' )
	except:
		pass
		
	try:
		os.remove( file_path + '/' + str(new_node)+'_1.hdf5' )
	except:
		pass
	
	wdh = wdh_log( file_path, max_size_str='36m' )
	wdh.add_one_node()
	
	num = 10**4
	for i in range( num ):
		g = 'user_' + str( random.randint(0,2**128) )
		v = str( i )
		bh = wdh.belong_to_hash( g )		
		wdh.write( g, 'd', [v] )

	wdh.add_one_node()
	
	'''
	f = h5py.File( file_path + '/' + str(old_node)+'_1.hdf5' )
	print( '--', f.keys() )
	for g in f.keys():
		del f[g]
	f.close()
	
	wdh.rewrite( str(old_node)+'_1.hdf5' )

	f = h5py.File( file_path + '/' + str(old_node)+'_1.hdf5', 'r' )
	print( f.keys() )
	f.close()
	'''
	
	'''
	mid_gs, mid_vs = [], []
	for f in os.listdir( wdh.file_path ):
		f_path = os.path.join( wdh.file_path, f )
		f = h5py.File( f_path, 'r' )
		for k in f.keys():
			mid_gs.append( k )
		f.close()
	
	print( set(gs)-set(mid_gs) )
	print( set(mid_gs) - set(gs) )
	'''
	
	'''
	# merge test
	wdh = wdh_log( file_path, max_size_str='2g' )
	wdh.merge( old_node )
	'''