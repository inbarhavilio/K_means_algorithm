from setuptools import setup, find_packages, Extension

###############################################
# Setup script to create the kmeans.c module  #
# to import to the python script              #
# Use for Python - C API implementation       #
###############################################

setup(
	name='mykmeanssp',  # name of module
	version='0.1.0',
	author='Inbar Havilio & Yuval Bloom',  # name of creators
	author_email='yuvalbloom@mail.tau.ac.il',
	description='K Means clustering algorithm C implementation', # description of C module
	install_requires=['numpy','scipy'],
	packages=find_packages(),

	license ='GPL-2',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'License :: OSI Approved :: Python Software Foundation License',
		'Programming Language :: Python :: implementation :: CPython',
	],

	ext_modules=[
		Extension(
			'mykmeanssp', # api to kmean.c file
			['kmeans.c'],
		),
	]
)